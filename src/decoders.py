"""Decoders shared across notebooks: a ridge regression baseline and a Kalman filter."""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RIDGE_ALPHAS = np.logspace(-1, 5, 13)


def fit_ridge(X, y, groups, alphas=RIDGE_ALPHAS, n_splits=5):
    """Ridge regression with the penalty (alpha) chosen by cross-validation grouped by trial.

    Returns the fitted GridSearchCV object: use .predict(), .best_score_ (cross-validated R²),
    and .best_params_["ridge__alpha"].
    """
    search = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": alphas},
                          cv=GroupKFold(n_splits=n_splits), scoring="r2", n_jobs=-1)
    search.fit(X, y, groups=groups)
    return search


class KalmanDecoder:
    """Kalman filter decoder, as used in classic motor-cortex brain-computer interface work.

    Two linear models, both fit by least squares on training trials:
      movement model:  x_t = A x_(t-1) + noise (covariance W)   how the hand state evolves
      neural model:    z_t = H x_t     + noise (covariance Q)   how neurons reflect the state
    x is the hand state [pos_x, pos_y, vel_x, vel_y, 1] and z is the neural observation.

    At each time step the filter predicts the next state from the movement model, then
    corrects that prediction using the new spikes, weighting each source by how reliable
    it has been.

    noise : how the neural noise covariance Q is estimated
        "full"      every neuron pair's covariance (about n x n numbers; the original)
        "diagonal"  each neuron's own variance only (n numbers; neurons treated as independent)
        "shrinkage" full covariance shrunk toward a simpler target (Ledoit-Wolf), with the
                    amount of shrinkage chosen automatically from the data
    """

    def __init__(self, q_reg=1e-3, noise="full"):
        if noise not in ("full", "diagonal", "shrinkage"):
            raise ValueError("noise must be 'full', 'diagonal', or 'shrinkage'")
        self.q_reg = q_reg  # small amount added to Q so it can always be inverted
        self.noise = noise

    @staticmethod
    def _valid(a):
        return ~np.isnan(a).any(axis=1)

    def fit(self, seqs):
        prev, nxt, xs, zs, firsts = [], [], [], [], []
        for s in seqs:
            X, Z = s["X"], s["Z"]
            pair = self._valid(X[:-1]) & self._valid(X[1:])   # consecutive steps, both known
            prev.append(X[:-1][pair]); nxt.append(X[1:][pair])
            ok = self._valid(X) & self._valid(Z)
            xs.append(X[ok]); zs.append(Z[ok])
            if self._valid(X[:1])[0]:
                firsts.append(X[0])
        prev, nxt = np.vstack(prev), np.vstack(nxt)
        xs, zs = np.vstack(xs), np.vstack(zs)

        self.A = np.linalg.lstsq(prev, nxt, rcond=None)[0].T
        self.W = np.cov((nxt - prev @ self.A.T).T)
        self.H = np.linalg.lstsq(xs, zs, rcond=None)[0].T
        resid = zs - xs @ self.H.T
        if self.noise == "full":
            Q = np.cov(resid.T)
        elif self.noise == "diagonal":
            Q = np.diag(resid.var(axis=0, ddof=1))
        else:
            Q = LedoitWolf().fit(resid).covariance_
        self.Q = Q + self.q_reg * np.eye(zs.shape[1])
        firsts = np.array(firsts)
        self.x0 = firsts.mean(axis=0)                 # typical starting state
        self.P0 = np.cov(firsts.T)                    # and how much it varies
        return self

    def predict(self, Z):
        """Decode one trial. Z: (steps x neurons). Returns estimated states (steps x 5)."""
        A, W, H, Q = self.A, self.W, self.H, self.Q
        x, P = self.x0.copy(), self.P0.copy()
        out = np.empty((len(Z), len(x)))
        for i, z in enumerate(Z):
            if i > 0:                                  # 1. predict from the movement model
                x = A @ x
                P = A @ P @ A.T + W
            if not np.isnan(z).any():                  # 2. correct using the new spikes
                S = H @ P @ H.T + Q
                K = np.linalg.solve(S, H @ P).T        # Kalman gain = P H' S^-1
                x = x + K @ (z - H @ x)
                P = P - K @ H @ P
                P = (P + P.T) / 2
            out[i] = x
        return out
