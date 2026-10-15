"""Population-dynamics tools: reach directions, condition averages, PCA, and jPCA."""
import numpy as np
import pandas as pd

from src.data import BIN_MS


def reach_angles(trials):
    """Reach direction of each trial in radians (0 = rightward, pi/2 = upward), from the
    hand's net displacement between the first and last bin of the trial window."""
    tids = trials["trial_id"].to_numpy()
    pos = trials["hand_pos"][["x", "y"]].to_numpy(dtype=float)
    angles = {}
    for tid in np.unique(tids):
        p = pos[tids == tid]
        p = p[~np.isnan(p).any(axis=1)]
        if len(p) >= 2:
            dx, dy = p[-1] - p[0]
            angles[tid] = np.arctan2(dy, dx)
    return angles


def direction_labels(angles, n_bins=8, min_trials=5):
    """Group trials into n_bins reach-direction bins (45 degrees each for 8 bins).

    Returns (labels, centers): labels maps trial id -> bin index, keeping only bins with at
    least `min_trials` trials; centers maps bin index -> bin center in degrees.
    """
    bins = {tid: int((a + np.pi) / (2 * np.pi) * n_bins) % n_bins for tid, a in angles.items()}
    counts = pd.Series(bins).value_counts()
    keep = set(counts[counts >= min_trials].index)
    labels = {tid: b for tid, b in bins.items() if b in keep}
    width = 360 / n_bins
    centers = {b: -180 + width / 2 + width * b for b in sorted(keep)}
    return labels, centers


def condition_average(trials, labels, signal="spikes_smth", bin_ms=BIN_MS):
    """Average firing rate (Hz) of every neuron at each time point, for each condition.

    labels maps trial id -> condition; trials without a label are skipped. Only time points
    that every condition has data for are kept.
    Returns rates (conditions x times x neurons), the condition list, and times (ms from
    movement onset).
    """
    t = (trials["align_time"].dt.total_seconds() * 1000).round().astype(int).to_numpy()
    tids = trials["trial_id"].to_numpy()
    cond = pd.Series(tids).map(labels).to_numpy()
    keep = ~pd.isna(cond)
    df = pd.DataFrame(trials[signal].to_numpy(dtype=float)[keep] / (bin_ms / 1000))
    df["cond"], df["t"] = cond[keep], t[keep]
    avg = df.groupby(["cond", "t"]).mean()
    conds = sorted(avg.index.get_level_values(0).unique())
    times = sorted(set.intersection(*[set(avg.loc[c].index) for c in conds]))
    rates = np.stack([avg.loc[c].loc[times].to_numpy() for c in conds])
    return rates, conds, np.array(times)


def soft_normalize(rates, soft_hz=5.0):
    """Scale each neuron by (its range of average rates + soft_hz), so strongly firing neurons
    don't dominate PCA, while nearly silent neurons aren't blown up into noise."""
    rng = np.nanmax(rates, axis=(0, 1)) - np.nanmin(rates, axis=(0, 1))
    return rates / (rng + soft_hz)


def pca(X):
    """PCA by singular value decomposition. X: (samples x features).

    Returns (components, explained_ratio, mean): components are rows (PC1 first), and
    explained_ratio is each PC's share of the total variance.
    """
    mean = X.mean(axis=0)
    _, s, vt = np.linalg.svd(X - mean, full_matrices=False)
    var = s ** 2
    return vt, var / var.sum(), mean


def _r2(target, pred):
    return 1 - np.sum((target - pred) ** 2) / np.sum((target - target.mean(axis=0)) ** 2)


def fit_jpca(rates, times, window, n_pcs=6, bin_ms=BIN_MS):
    """jPCA: find the plane where the population state rotates most consistently.

    Steps: keep the time window; subtract the average across conditions at each time point
    (removing activity shared by all reaches); reduce to the top n_pcs principal components;
    then fit how the state changes from one bin to the next (dX = M x) two ways:
      - M_best: any linear dynamics (unconstrained)
      - M_skew: pure rotation (skew-symmetric matrix, M = -M transpose)
    The plane of strongest rotation comes from M_skew's largest-magnitude eigenvalue pair.

    Returns a dict with the projected trajectories ("proj": conditions x times x 2), the fit
    R² values, the rotation frequency (Hz), the share of variance in the jPC plane, and times.
    """
    sel = (times >= window[0]) & (times <= window[1])
    A = rates[:, sel, :]
    A = A - A.mean(axis=0, keepdims=True)                  # remove the cross-condition mean
    C, T, N = A.shape
    comps, ratio, mean = pca(A.reshape(C * T, N))
    X = ((A.reshape(C * T, N) - mean) @ comps[:n_pcs].T).reshape(C, T, n_pcs)

    states = X[:, :-1].reshape(-1, n_pcs)                  # state at each step
    deriv = (X[:, 1:] - X[:, :-1]).reshape(-1, n_pcs)      # change to the next step

    M_best = np.linalg.lstsq(states, deriv, rcond=None)[0].T

    pairs = [(i, j) for i in range(n_pcs) for j in range(i + 1, n_pcs)]
    basis = []
    for i, j in pairs:
        B = np.zeros((n_pcs, n_pcs)); B[i, j], B[j, i] = 1.0, -1.0
        basis.append(B)
    design = np.column_stack([(states @ B.T).ravel() for B in basis])
    coef = np.linalg.lstsq(design, deriv.ravel(), rcond=None)[0]
    M_skew = sum(c * B for c, B in zip(coef, basis))

    eigvals, eigvecs = np.linalg.eig(M_skew)
    top = np.argmax(np.abs(eigvals.imag))
    v = eigvecs[:, top]
    plane, _ = np.linalg.qr(np.column_stack([v.real, v.imag]))
    proj = X @ plane                                       # conditions x times x 2

    flat = X.reshape(-1, n_pcs)
    centered = flat - flat.mean(axis=0)
    in_plane = np.sum((centered @ plane) ** 2) / np.sum(centered ** 2)

    return {"proj": proj, "times": times[sel],
            "r2_skew": _r2(deriv, states @ M_skew.T), "r2_best": _r2(deriv, states @ M_best.T),
            "freq_hz": abs(eigvals[top].imag) / (2 * np.pi) / (bin_ms / 1000),
            "plane_variance": in_plane, "pcs_variance": ratio[:n_pcs].sum()}
