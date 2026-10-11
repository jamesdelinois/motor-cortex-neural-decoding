"""Reusable steps for loading MC_Maze and building decoder inputs.

Notebooks import these functions so every phase uses the same loading,
cleaning, and feature-building code.
"""
import numpy as np
import pandas as pd

BIN_MS = 20  # width of each time bin, in milliseconds


def load_dataset(path, bin_ms=BIN_MS):
    """Load the MC_Maze training file, repair its time index, and bin the spikes.

    Why the repair is needed: nlb_tools combines the spike and behavior signals
    into one table. They were recorded on slightly different sets of timestamps,
    and with newer versions of pandas the combined table comes out partly out of
    order (timestamps missing from the first signal get appended at the end).
    Rounding to whole milliseconds and grouping by timestamp sorts the table back
    into a regular 1 ms grid without losing any rows.
    """
    from nlb_tools.nwb_interface import NWBDataset  # imported here so the other functions work without it

    dataset = NWBDataset(path, "*train", split_heldout=False)
    dataset.data.index = dataset.data.index.round("ms")
    dataset.data = dataset.data.groupby(level=0).first()
    dataset.data.index.name = "clock_time"
    dataset.resample(bin_ms)
    return dataset


def get_trials(dataset, history_ms, window=(-250, 450)):
    """Cut every trial to the decoding window around movement onset.

    `window` is in ms relative to movement onset; (-250, 450) is the window the
    Neural Latents Benchmark uses to score velocity decoding. `history_ms` adds
    extra data before the window (marked in the `margin` column) so that even
    the first bin in the window has a full spike history.
    """
    return dataset.make_trial_data(
        align_field="move_onset_time", align_range=window, margin=history_ms
    )


def build_features(trials, n_lags, bin_ms=BIN_MS):
    """Turn trial data into a decoder's inputs (X) and target (y).

    Each row of X is one time bin: the spike counts of every neuron in that bin,
    followed by the counts from each of the `n_lags` bins before it. Each row
    of y is the hand velocity (x, y) in that same bin.

    Returns
    -------
    X : array, shape (bins, neurons * (n_lags + 1))
    y : array, shape (bins, 2)
    groups : array, shape (bins,)  trial id of each bin, used to split by trial
    t_ms : array, shape (bins,)    time of each bin relative to movement onset
    """
    t_all = (trials["align_time"].dt.total_seconds() * 1000).round().astype(int).to_numpy()
    spikes = trials["spikes"].to_numpy(dtype=float)
    vel = trials["hand_vel"][["x", "y"]].to_numpy(dtype=float)
    trial_ids = trials["trial_id"].to_numpy()
    in_window = ~trials["margin"].to_numpy(dtype=bool)

    X_parts, y_parts, g_parts, t_parts = [], [], [], []
    for tid in np.unique(trial_ids):
        rows = np.flatnonzero(trial_ids == tid)
        # This trial's spikes, looked up by time (ms from movement onset)
        S = pd.DataFrame(spikes[rows], index=t_all[rows])
        target_rows = rows[in_window[rows]]
        t_target = t_all[target_rows]
        # Spike counts k bins earlier, for k = 0 (the current bin), 1, 2, ...
        lagged = [S.reindex(t_target - k * bin_ms).to_numpy() for k in range(n_lags + 1)]
        X_parts.append(np.hstack(lagged))
        y_parts.append(vel[target_rows])
        g_parts.append(np.full(len(target_rows), tid))
        t_parts.append(t_target)

    X = np.vstack(X_parts)
    y = np.vstack(y_parts)
    groups = np.concatenate(g_parts)
    t_ms = np.concatenate(t_parts)

    # Drop any bin with missing values (for example, a history bin that isn't there)
    ok = ~np.isnan(X).any(axis=1) & ~np.isnan(y).any(axis=1)
    return X[ok], y[ok], groups[ok], t_ms[ok]
