"""sweep.py — HMM model body and named-experiment runners (iteration 2).

This is the autoresearch loop's playground (per program.md). The harness in
harness.py is fixed; this file is freely editable.

v0 ships only `exp_001_baseline` (K=3 Gaussian on rolling-200 drawdown,
homogeneous transitions) so the loop can establish the baseline TSV row.
Subsequent versions add multivariate Gaussian + NH-HMM transitions for
exp_002_proposal_k3 and the Phase 2 ablations.

Run:
    python sweep.py --experiment exp_001_baseline

The HMM core (forward/forward-backward/Viterbi/Baum-Welch) is lifted from
the validated notebook code in `notebooks/regime_drawdown_2022.org`. The
fit/score discipline (rolling-200 drawdown feature, μ-sort labelling,
5 k-means jittered restarts) mirrors that notebook so exp_001 reproduces
its result on the extended 2017-2024 window under the new scoring.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import digamma, logsumexp
from scipy.stats import t as student_t
from sklearn.cluster import KMeans

import harness


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

DRAWDOWN_WINDOW = 200  # rolling-window max length for the drawdown feature


def rolling_drawdown(log_price: np.ndarray, window: int = DRAWDOWN_WINDOW) -> np.ndarray:
    """d_t = log(p_t) − max_{s ∈ [t-window+1, t]} log(p_s).

    NaN for the first `window-1` rows (rolling max not fully populated).
    Causal by construction.
    """
    s = pd.Series(log_price)
    rolling_max = s.rolling(window=window, min_periods=window).max()
    return (s - rolling_max).to_numpy()


# ---------------------------------------------------------------------------
# HMM core — log-space forward / forward-backward / Viterbi
# ---------------------------------------------------------------------------

def forward(log_pi: np.ndarray, log_A: np.ndarray, log_B: np.ndarray) -> np.ndarray:
    """Causal forward pass. Returns log_alpha (T, K)."""
    T, K = log_B.shape
    log_alpha = np.full((T, K), -np.inf)
    log_alpha[0] = log_pi + log_B[0]
    for t in range(1, T):
        log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0) + log_B[t]
    return log_alpha


def forward_backward(log_pi, log_A, log_B):
    """Smoothed posteriors + xi for Baum-Welch."""
    T, K = log_B.shape
    log_alpha = forward(log_pi, log_A, log_B)
    log_beta = np.full((T, K), -np.inf)
    log_beta[T - 1] = 0.0
    for t in range(T - 2, -1, -1):
        log_beta[t] = logsumexp(log_A + log_B[t + 1][None, :] + log_beta[t + 1][None, :], axis=1)
    log_likelihood = logsumexp(log_alpha[T - 1])
    log_gamma = log_alpha + log_beta - log_likelihood
    log_xi = (
        log_alpha[:-1, :, None]
        + log_A[None, :, :]
        + log_B[1:, None, :]
        + log_beta[1:, None, :]
        - log_likelihood
    )
    return log_gamma, log_xi, log_likelihood


# ---------------------------------------------------------------------------
# Emissions — Gaussian (used by baseline)
# ---------------------------------------------------------------------------

def gaussian_log_emissions(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    z = (x[:, None] - mu[None, :]) / sigma[None, :]
    return -0.5 * np.log(2 * np.pi) - np.log(sigma)[None, :] - 0.5 * z * z


def m_step_gaussian(x, log_gamma):
    gamma = np.exp(log_gamma)
    Nk = gamma.sum(axis=0)
    mu = (gamma * x[:, None]).sum(axis=0) / Nk
    diff_sq = (x[:, None] - mu[None, :]) ** 2
    sigma = np.sqrt((gamma * diff_sq).sum(axis=0) / Nk) + 1e-6
    return mu, sigma


# ---------------------------------------------------------------------------
# Init + transition M-step
# ---------------------------------------------------------------------------

def kmeans_init(x, K, rng, jitter=0.0):
    # n_init=3 is enough on 1D data — the algorithm is essentially deterministic
    # and the restart-diversity comes from `jitter` applied to the cluster
    # means rather than KMeans's internal restarts.
    km = KMeans(n_clusters=K, n_init=3, random_state=int(rng.integers(1 << 31)))
    labels = km.fit_predict(x.reshape(-1, 1))
    mu = np.array([x[labels == i].mean() for i in range(K)])
    sigma = np.array([x[labels == i].std(ddof=0) + 1e-6 for i in range(K)])
    pi = np.array([(labels == i).mean() for i in range(K)])
    if jitter > 0:
        mu = mu + rng.normal(0, jitter * x.std(), size=K)
    # Sticky prior — let EM concentrate the diagonal if data supports it.
    A = np.full((K, K), 0.05 / (K - 1)) if K > 1 else np.ones((1, 1))
    np.fill_diagonal(A, 0.95)
    return mu, sigma, pi, A


def m_step_init_trans(log_gamma, log_xi):
    log_pi = log_gamma[0] - logsumexp(log_gamma[0])
    log_A = logsumexp(log_xi, axis=0) - logsumexp(log_xi, axis=(0, 2))[:, None]
    return log_pi, log_A


# ---------------------------------------------------------------------------
# Baum-Welch driver + restart wrapper (homogeneous transitions)
# ---------------------------------------------------------------------------

@dataclass
class HMMFit:
    log_pi: np.ndarray
    log_A: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    log_likelihood: float
    n_iter: int
    converged: bool
    family: str


def baum_welch_gaussian(x, K, rng=None, max_iter=200, tol=1e-5, jitter=0.0,
                        init: Optional["HMMFit"] = None):
    """EM for K-state Gaussian HMM. Optionally warm-start from a prior fit.

    When `init` is provided, skip k-means and start EM from those params —
    typically converges in 1-3 iterations. When `init` is None, fall back to
    k-means init with optional jitter for restart diversity.
    """
    if init is not None:
        mu = init.mu.copy()
        sigma = init.sigma.copy()
        log_pi = init.log_pi.copy()
        log_A = init.log_A.copy()
    else:
        assert rng is not None, "rng required for cold-start k-means init"
        mu, sigma, pi, A = kmeans_init(x, K, rng, jitter=jitter)
        log_pi = np.log(pi + 1e-12)
        log_A = np.log(A + 1e-12)

    prev_ll = -np.inf
    converged = False
    it = 0
    for it in range(max_iter):
        log_B = gaussian_log_emissions(x, mu, sigma)
        log_gamma, log_xi, ll = forward_backward(log_pi, log_A, log_B)
        log_pi, log_A = m_step_init_trans(log_gamma, log_xi)
        mu, sigma = m_step_gaussian(x, log_gamma)
        if abs(ll - prev_ll) < tol:
            converged = True
            it += 1
            break
        prev_ll = ll
    return HMMFit(log_pi, log_A, mu, sigma, ll, it, converged, "gaussian")


def fit_with_restarts(x, K, n_restarts=5, seed=20260603, family="gaussian",
                      init: Optional["HMMFit"] = None):
    """Cold-start with k-means + restarts, OR warm-start from `init` (single
    EM run, no restarts since the init is already near a local optimum)."""
    if init is not None:
        if family != "gaussian":
            raise NotImplementedError(f"warm-start for family={family} not in v0")
        return baum_welch_gaussian(x, K, init=init)

    best = None
    master = np.random.default_rng(seed)
    for r in range(n_restarts):
        rng = np.random.default_rng(master.integers(1 << 31))
        jitter = 0.0 if r == 0 else 0.15
        if family == "gaussian":
            fit = baum_welch_gaussian(x, K, rng=rng, jitter=jitter)
        else:
            raise NotImplementedError(f"family={family} not in v0")
        if best is None or fit.log_likelihood > best.log_likelihood:
            best = fit
    if best is None:
        raise RuntimeError("all restarts failed")
    return best


# ---------------------------------------------------------------------------
# Baseline model — wraps the fit + the forward-posterior projection
# ---------------------------------------------------------------------------

class BaselineModel:
    """K=3 Gaussian on rolling-200 drawdown, homogeneous transitions.

    forward_posterior(data) returns (T, 3) posteriors with columns in canonical
    (bear, ranging, bull) order — μ-sort: deepest drawdown = bear, shallowest =
    bull. NaN rows for warmup days where the rolling-200 drawdown is undefined.
    """

    def __init__(self, fit: HMMFit):
        self.fit = fit
        # μ-sort: state with lowest μ (deepest drawdown) → bear (col 0),
        # state with highest μ (shallowest drawdown) → bull (col 2).
        self.order = np.argsort(fit.mu)
        assert len(self.order) == 3, "Baseline expects K=3"

    def forward_posterior(self, data: pd.DataFrame) -> np.ndarray:
        log_price = data["log_price"].to_numpy()
        feat = rolling_drawdown(log_price, window=DRAWDOWN_WINDOW)
        valid = ~np.isnan(feat)

        if valid.sum() < 2:  # need at least one transition for forward pass
            return np.full((len(data), 3), np.nan)

        obs = feat[valid]
        log_B = gaussian_log_emissions(obs, self.fit.mu, self.fit.sigma)
        log_alpha = forward(self.fit.log_pi, self.fit.log_A, log_B)
        log_post = log_alpha - logsumexp(log_alpha, axis=1, keepdims=True)
        post = np.exp(log_post)                # (T_valid, K)
        post_ordered = post[:, self.order]      # (T_valid, 3) in (bear, ranging, bull)

        full = np.full((len(data), 3), np.nan)
        full[valid] = post_ordered
        return full


def make_baseline_factory(K: int = 3, n_restarts_cold: int = 5):
    """Stateful factory: first call does a cold-start k-means + jittered
    restarts; subsequent calls warm-start from the previous fit (single EM
    run that typically converges in 1-3 iterations).

    Warm-start is what makes the walk-forward tractable — without it, every
    refit re-runs k-means + N restarts on growing data, which scales badly.
    """
    state = {"last_fit": None}

    def factory(train_data: pd.DataFrame) -> BaselineModel:
        log_price = train_data["log_price"].to_numpy()
        feat = rolling_drawdown(log_price, window=DRAWDOWN_WINDOW)
        obs = feat[~np.isnan(feat)]
        if len(obs) < 50:
            raise RuntimeError(f"Not enough valid observations to fit: {len(obs)}")

        if state["last_fit"] is None:
            fit = fit_with_restarts(obs, K=K, n_restarts=n_restarts_cold, family="gaussian")
        else:
            fit = fit_with_restarts(obs, K=K, family="gaussian", init=state["last_fit"])

        state["last_fit"] = fit
        return BaselineModel(fit)
    return factory


# ---------------------------------------------------------------------------
# Named-experiment runners
# ---------------------------------------------------------------------------

def run_exp_001_baseline() -> None:
    """K=3 Gaussian HMM on rolling-200-day drawdown, homogeneous transitions.

    The current best from regime_drawdown_2022.org, re-evaluated on the
    extended 2017-2024 window with the new misclassification scoring.
    """
    factory = make_baseline_factory(K=3, n_restarts_cold=5)
    harness.evaluate(
        experiment_id="exp_001_baseline",
        model_factory=factory,
        observation="drawdown_200",
        K=3,
        emission="gaussian",
        transitions="homogeneous",
        comment="Baseline: K=3 Gaussian on rolling-200 drawdown, mu-sort labelling. From regime_drawdown_2022.org.",
    )


EXPERIMENTS = {
    "exp_001_baseline": run_exp_001_baseline,
    # exp_002_proposal_k3 added in v1 (multivariate Gaussian + NH-HMM)
    # exp_003 / exp_004 added in Phase 2 based on Phase 1 outcome
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=list(EXPERIMENTS.keys()))
    args = parser.parse_args()
    EXPERIMENTS[args.experiment]()


if __name__ == "__main__":
    sys.exit(main())
