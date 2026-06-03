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
from scipy.linalg import solve_triangular
from scipy.optimize import brentq, minimize
from scipy.special import digamma, gammaln, logsumexp
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


REALISED_VOL_WINDOW = 5  # paper-2 "5-day realised vol" channel for the proposal


def realised_vol(log_return: np.ndarray, window: int = REALISED_VOL_WINDOW) -> np.ndarray:
    """σ_t^{Wd} = rolling std of daily log-returns over the trailing `window`.

    NaN for the first `window-1` rows. Causal (uses only past+current returns).
    The first element of `log_return` is itself NaN (diff of log_price), so the
    rolling window propagates that until it slides past day 0.
    """
    s = pd.Series(log_return)
    return s.rolling(window=window, min_periods=window).std(ddof=0).to_numpy()


def trivariate_features(data: pd.DataFrame):
    """Build the proposal's (rₜ, σₜ^{5d}, dₜ_{200}) observation matrix.

    Returns (F, valid) where F is (T, 3) with columns [log_return,
    realised_vol_5d, drawdown_200] and `valid` is the boolean mask of rows where
    all three channels are populated (drawdown's 200-day warmup dominates).
    """
    log_price = data["log_price"].to_numpy()
    log_return = data["log_return"].to_numpy()
    r = log_return
    vol = realised_vol(log_return, REALISED_VOL_WINDOW)
    dd = rolling_drawdown(log_price, DRAWDOWN_WINDOW)
    F = np.column_stack([r, vol, dd])
    valid = ~np.isnan(F).any(axis=1)
    return F, valid


def drawdown_only_features(data: pd.DataFrame):
    """Univariate rolling-200 drawdown observation (the baseline's channel),
    shaped (T, 1) for the multivariate code path. Used by exp_004's NH-only
    ablation so it shares the proposal's transition machinery on the *baseline's*
    observation."""
    dd = rolling_drawdown(data["log_price"].to_numpy(), DRAWDOWN_WINDOW)
    F = dd[:, None]
    valid = ~np.isnan(F).any(axis=1)
    return F, valid


def vix_covariate(data: pd.DataFrame) -> np.ndarray:
    """Raw VIX close aligned to `data`'s rows, with leading/embedded NaN handled.

    The harness forward-fills VIX onto the crypto calendar; any residual NaN
    (none expected since VIX and BTC share a start date) is forward/back-filled
    so the covariate is always finite. Standardisation happens in the model
    factory using *training-window* statistics (causal)."""
    vix = pd.Series(data["vix_close"].to_numpy()).ffill().bfill().to_numpy()
    return vix


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


# ===========================================================================
# Proposal machinery (exp_002+): multivariate Gaussian emissions + NH-HMM
# transitions with a VIX covariate.
# ===========================================================================

# ---------------------------------------------------------------------------
# Multivariate Gaussian emissions (full Σ) + M-step
# ---------------------------------------------------------------------------

def mvn_log_emissions(X: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """Log N(x_t | μ_k, Σ_k) for every (t, k). Full covariance, Cholesky-based.

    X: (T, D)   mu: (K, D)   Sigma: (K, D, D)  ->  log_B: (T, K)
    """
    T, D = X.shape
    K = mu.shape[0]
    log_B = np.empty((T, K))
    const = D * np.log(2 * np.pi)
    for k in range(K):
        L = np.linalg.cholesky(Sigma[k])              # (D, D) lower
        diff = (X - mu[k]).T                           # (D, T)
        z = solve_triangular(L, diff, lower=True)      # (D, T)
        maha = np.sum(z * z, axis=0)                   # (T,)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        log_B[:, k] = -0.5 * (const + logdet + maha)
    return log_B


def m_step_mvn(X: np.ndarray, log_gamma: np.ndarray, reg: float = 1e-4):
    """Weighted MLE for K full-covariance Gaussians. Returns (mu, Sigma)."""
    gamma = np.exp(log_gamma)                          # (T, K)
    Nk = gamma.sum(axis=0) + 1e-12                     # (K,)
    mu = (gamma.T @ X) / Nk[:, None]                   # (K, D)
    D = X.shape[1]
    K = mu.shape[0]
    Sigma = np.empty((K, D, D))
    eye = np.eye(D)
    for k in range(K):
        diff = X - mu[k]                               # (T, D)
        S = (gamma[:, k:k + 1] * diff).T @ diff / Nk[k]
        Sigma[k] = S + reg * eye                       # ridge for conditioning
    return mu, Sigma


# ---------------------------------------------------------------------------
# Non-homogeneous transitions: A_t[i,j] = softmax_j( W[i,j] · x_t )
#   x_t = (1, vix_std_t)   reference column j=0 fixed at 0 (identifiability)
# ---------------------------------------------------------------------------

def nh_log_A_seq(W: np.ndarray, Xc: np.ndarray) -> np.ndarray:
    """Time-varying log transition matrices.

    W: (K, K, Dc) with W[:, 0, :] held at 0.  Xc: (T, Dc).
    Returns log_A_seq: (T, K, K) where log_A_seq[t, i, j] is the log-prob of the
    transition INTO step t (from state i at t-1 to state j at t), driven by the
    covariate x_t. Row t=0 is computed but unused (t=0 uses the initial dist).
    """
    s = np.einsum("ijd,td->tij", W, Xc)                # (T, K, K)
    return s - logsumexp(s, axis=2, keepdims=True)


def forward_nh(log_pi, log_A_seq, log_B):
    """Causal forward pass with time-varying transitions. Returns log_alpha."""
    T, K = log_B.shape
    log_alpha = np.full((T, K), -np.inf)
    log_alpha[0] = log_pi + log_B[0]
    for t in range(1, T):
        log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_A_seq[t], axis=0) + log_B[t]
    return log_alpha


def forward_backward_nh(log_pi, log_A_seq, log_B):
    """Smoothed posteriors + xi for the non-homogeneous Baum-Welch."""
    T, K = log_B.shape
    log_alpha = forward_nh(log_pi, log_A_seq, log_B)
    log_beta = np.full((T, K), -np.inf)
    log_beta[T - 1] = 0.0
    for t in range(T - 2, -1, -1):
        log_beta[t] = logsumexp(
            log_A_seq[t + 1] + log_B[t + 1][None, :] + log_beta[t + 1][None, :], axis=1
        )
    log_likelihood = logsumexp(log_alpha[T - 1])
    log_gamma = log_alpha + log_beta - log_likelihood
    # xi[t] is the transition t -> t+1, governed by log_A_seq[t+1]; shape (T-1,K,K)
    log_xi = (
        log_alpha[:-1, :, None]
        + log_A_seq[1:, :, :]
        + log_B[1:, None, :]
        + log_beta[1:, None, :]
        - log_likelihood
    )
    return log_gamma, log_xi, log_likelihood


def nh_transition_mstep(log_xi, Xc, W_init, maxiter: int = 50):
    """M-step for softmax transitions: one weighted multinomial logistic
    regression per 'from' state i, maximising the expected complete-data
    log-likelihood Σ_t Σ_j ξ_t[i,j] · log softmax_j(W_i·x_t).

    log_xi: (T-1, K, K)   Xc: (T, Dc)  -> W: (K, K, Dc) with col 0 == 0.
    Warm-started from W_init for fast convergence across refits.
    """
    xi = np.exp(log_xi)                                 # (T-1, K, K)
    Xt = Xc[1:]                                          # (T-1, Dc) covariate per transition
    Tm1, Dc = Xt.shape
    K = xi.shape[1]
    W = W_init.copy()

    for i in range(K):
        target = xi[:, i, :]                            # (T-1, K)
        total = target.sum(axis=1)                      # (T-1,) = gamma_src at t (== marginal)
        # Free parameters: rows j=1..K-1 (j=0 is the reference, fixed 0).
        def neg_ll_grad(theta):
            Wi = np.zeros((K, Dc))
            Wi[1:] = theta.reshape(K - 1, Dc)
            s = Xt @ Wi.T                               # (T-1, K)
            logp = s - logsumexp(s, axis=1, keepdims=True)
            nll = -np.sum(target * logp)
            p = np.exp(logp)                            # (T-1, K)
            # grad wrt W[j,:] = Σ_t (total_t * p_tj - target_tj) * x_t
            g = ((total[:, None] * p - target).T @ Xt)  # (K, Dc)
            return nll, g[1:].ravel()

        theta0 = W_init[i, 1:, :].ravel()
        res = minimize(neg_ll_grad, theta0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter})
        W[i, 0, :] = 0.0
        W[i, 1:, :] = res.x.reshape(K - 1, Dc)
    return W


# ---------------------------------------------------------------------------
# NH-MVN Baum-Welch driver + restart wrapper
# ---------------------------------------------------------------------------

@dataclass
class NHMVNFit:
    log_pi: np.ndarray          # (K,)
    W: np.ndarray               # (K, K, Dc) softmax transition weights
    mu: np.ndarray              # (K, D)
    Sigma: np.ndarray           # (K, D, D)
    log_likelihood: float
    n_iter: int
    converged: bool
    family: str = "mvn_nh"


def _sticky_W_init(A0: np.ndarray, Dc: int) -> np.ndarray:
    """Encode a homogeneous sticky matrix A0 as softmax intercepts (slope=0)."""
    K = A0.shape[0]
    W = np.zeros((K, K, Dc))
    logA0 = np.log(A0 + 1e-12)
    # Reference column j=0 -> subtract its score so col-0 stays 0.
    W[:, :, 0] = logA0 - logA0[:, 0:1]
    W[:, 0, 0] = 0.0
    return W


def kmeans_init_mvn(X, K, rng, jitter=0.0):
    km = KMeans(n_clusters=K, n_init=3, random_state=int(rng.integers(1 << 31)))
    labels = km.fit_predict(X)
    D = X.shape[1]
    mu = np.array([X[labels == i].mean(axis=0) for i in range(K)])
    Sigma = np.empty((K, D, D))
    eye = np.eye(D)
    for i in range(K):
        Xi = X[labels == i]
        if len(Xi) > D:
            Sigma[i] = np.cov(Xi.T) + 1e-3 * eye
        else:
            Sigma[i] = eye
    pi = np.array([(labels == i).mean() for i in range(K)])
    if jitter > 0:
        mu = mu + rng.normal(0, jitter, size=mu.shape) * X.std(axis=0)[None, :]
    return mu, Sigma, pi


def baum_welch_nh_mvn(X, Xc, K, rng=None, max_iter=120, tol=1e-4, jitter=0.0,
                      trans_maxiter=50, init: Optional["NHMVNFit"] = None):
    """EM for a K-state multivariate-Gaussian NH-HMM. Warm-start from `init`."""
    Dc = Xc.shape[1]
    if init is not None:
        mu = init.mu.copy()
        Sigma = init.Sigma.copy()
        log_pi = init.log_pi.copy()
        W = init.W.copy()
    else:
        assert rng is not None, "rng required for cold-start init"
        mu, Sigma, pi, = (*kmeans_init_mvn(X, K, rng, jitter=jitter),)
        log_pi = np.log(pi + 1e-12)
        A0 = np.full((K, K), 0.05 / (K - 1))
        np.fill_diagonal(A0, 0.95)
        W = _sticky_W_init(A0, Dc)

    prev_ll = -np.inf
    converged = False
    it = 0
    for it in range(max_iter):
        log_B = mvn_log_emissions(X, mu, Sigma)
        log_A_seq = nh_log_A_seq(W, Xc)
        log_gamma, log_xi, ll = forward_backward_nh(log_pi, log_A_seq, log_B)
        # M-step
        log_pi = log_gamma[0] - logsumexp(log_gamma[0])
        W = nh_transition_mstep(log_xi, Xc, W, maxiter=trans_maxiter)
        mu, Sigma = m_step_mvn(X, log_gamma)
        if np.isfinite(ll) and abs(ll - prev_ll) < tol:
            converged = True
            it += 1
            break
        prev_ll = ll
    return NHMVNFit(log_pi, W, mu, Sigma, ll, it, converged)


def fit_nh_mvn_with_restarts(X, Xc, K, n_restarts=3, seed=20260603,
                             init: Optional["NHMVNFit"] = None):
    if init is not None:
        return baum_welch_nh_mvn(X, Xc, K, init=init, max_iter=40)
    best = None
    master = np.random.default_rng(seed)
    for r in range(n_restarts):
        rng = np.random.default_rng(master.integers(1 << 31))
        jitter = 0.0 if r == 0 else 0.25
        fit = baum_welch_nh_mvn(X, Xc, K, rng=rng, jitter=jitter)
        if best is None or (np.isfinite(fit.log_likelihood) and
                            fit.log_likelihood > best.log_likelihood):
            best = fit
    if best is None:
        raise RuntimeError("all NH-MVN restarts failed")
    return best


# ---------------------------------------------------------------------------
# Homogeneous MVN Baum-Welch (exp_003 ablation: multivariate obs, fixed A)
# ---------------------------------------------------------------------------

@dataclass
class HomogMVNFit:
    log_pi: np.ndarray          # (K,)
    log_A: np.ndarray           # (K, K) homogeneous transitions
    mu: np.ndarray              # (K, D)
    Sigma: np.ndarray           # (K, D, D)
    log_likelihood: float
    n_iter: int
    converged: bool
    family: str = "mvn_homog"


def baum_welch_homog_mvn(X, K, rng=None, max_iter=120, tol=1e-4, jitter=0.0,
                         init: Optional["HomogMVNFit"] = None):
    """EM for a K-state multivariate-Gaussian HMM with homogeneous transitions.

    Same emission body and warm-start discipline as the NH driver, but the
    transition M-step is the closed-form normalised-xi update (no inner optimise)."""
    if init is not None:
        mu = init.mu.copy()
        Sigma = init.Sigma.copy()
        log_pi = init.log_pi.copy()
        log_A = init.log_A.copy()
    else:
        assert rng is not None, "rng required for cold-start init"
        mu, Sigma, pi = kmeans_init_mvn(X, K, rng, jitter=jitter)
        log_pi = np.log(pi + 1e-12)
        A0 = np.full((K, K), 0.05 / (K - 1))
        np.fill_diagonal(A0, 0.95)
        log_A = np.log(A0 + 1e-12)

    prev_ll = -np.inf
    converged = False
    it = 0
    for it in range(max_iter):
        log_B = mvn_log_emissions(X, mu, Sigma)
        log_gamma, log_xi, ll = forward_backward(log_pi, log_A, log_B)
        log_pi, log_A = m_step_init_trans(log_gamma, log_xi)
        mu, Sigma = m_step_mvn(X, log_gamma)
        if np.isfinite(ll) and abs(ll - prev_ll) < tol:
            converged = True
            it += 1
            break
        prev_ll = ll
    return HomogMVNFit(log_pi, log_A, mu, Sigma, ll, it, converged)


def fit_homog_mvn_with_restarts(X, K, n_restarts=3, seed=20260603,
                                init: Optional["HomogMVNFit"] = None):
    if init is not None:
        return baum_welch_homog_mvn(X, K, init=init, max_iter=40)
    best = None
    master = np.random.default_rng(seed)
    for r in range(n_restarts):
        rng = np.random.default_rng(master.integers(1 << 31))
        jitter = 0.0 if r == 0 else 0.25
        fit = baum_welch_homog_mvn(X, K, rng=rng, jitter=jitter)
        if best is None or (np.isfinite(fit.log_likelihood) and
                            fit.log_likelihood > best.log_likelihood):
            best = fit
    if best is None:
        raise RuntimeError("all homogeneous-MVN restarts failed")
    return best


# ---------------------------------------------------------------------------
# Multivariate Student-t emissions (fat tails, paper 3) + ECM M-step
# ---------------------------------------------------------------------------

NU_MIN = 2.05         # keep finite covariance (ν>2); below this the t has no var
NU_MAX = 200.0        # ν>=this is numerically Gaussian — clamp & treat as such


def _mahalanobis(X, mu, Sigma):
    """(x_t-μ)^T Σ^{-1} (x_t-μ) for all t, plus logdet(Σ). Cholesky-based."""
    L = np.linalg.cholesky(Sigma)
    diff = (X - mu).T                                  # (D, T)
    z = solve_triangular(L, diff, lower=True)          # (D, T)
    maha = np.sum(z * z, axis=0)                       # (T,)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return maha, logdet


def mvt_log_emissions(X, mu, Sigma, nu):
    """Log pdf of the D-variate Student-t per (t, k).

    X:(T,D)  mu:(K,D)  Sigma:(K,D,D) scale matrices  nu:(K,) dof  -> (T,K).
    """
    T, D = X.shape
    K = mu.shape[0]
    log_B = np.empty((T, K))
    for k in range(K):
        maha, logdet = _mahalanobis(X, mu[k], Sigma[k])
        nk = nu[k]
        log_B[:, k] = (
            gammaln((nk + D) / 2.0) - gammaln(nk / 2.0)
            - 0.5 * D * np.log(nk * np.pi)
            - 0.5 * logdet
            - 0.5 * (nk + D) * np.log1p(maha / nk)
        )
    return log_B


def _solve_nu(c, lo=NU_MIN, hi=NU_MAX):
    """Solve log(ν/2) − digamma(ν/2) + c = 0 for ν. The LHS is +∞ at ν→0 and
    →0⁺ as ν→∞, so a root exists iff c < 0; otherwise ν→∞ (Gaussian), clamp hi."""
    def f(nu):
        return np.log(nu / 2.0) - digamma(nu / 2.0) + c
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        # No sign change: pick the bound the function is heading toward.
        return hi if abs(fhi) < abs(flo) else lo
    return brentq(f, lo, hi, maxiter=100)


def m_step_mvt(X, log_gamma, mu, Sigma, nu, reg: float = 1e-4):
    """One ECM update for K multivariate-t states given HMM responsibilities.

    Uses the Peel-McLachlan latent-scale weights u_tk = (ν_k+D)/(ν_k+maha_tk),
    re-weighting the Gaussian-style mean/scale updates and solving the dof
    fixed-point per state. Returns (mu_new, Sigma_new, nu_new)."""
    gamma = np.exp(log_gamma)                          # (T, K)
    T, D = X.shape
    K = mu.shape[0]
    mu_new = np.empty_like(mu)
    Sigma_new = np.empty_like(Sigma)
    nu_new = np.empty_like(nu)
    eye = np.eye(D)
    for k in range(K):
        maha, _ = _mahalanobis(X, mu[k], Sigma[k])
        u = (nu[k] + D) / (nu[k] + maha)               # (T,) latent precisions
        g = gamma[:, k]
        gu = g * u
        Sgu = gu.sum() + 1e-12
        Nk = g.sum() + 1e-12
        mk = (gu[:, None] * X).sum(axis=0) / Sgu
        diff = X - mk
        Sk = (gu[:, None] * diff).T @ diff / Nk        # scale matrix (γu weighted, /Σγ)
        Sk = Sk + reg * eye
        mu_new[k] = mk
        Sigma_new[k] = Sk
        # dof fixed-point: constant term from current-iter u, ν.
        c = (1.0 + (g * (np.log(u) - u)).sum() / Nk
             + digamma((nu[k] + D) / 2.0) - np.log((nu[k] + D) / 2.0))
        nu_new[k] = _solve_nu(c)
    return mu_new, Sigma_new, nu_new


@dataclass
class HomogMVTFit:
    log_pi: np.ndarray
    log_A: np.ndarray
    mu: np.ndarray
    Sigma: np.ndarray           # scale matrices
    nu: np.ndarray              # (K,) per-state dof
    log_likelihood: float
    n_iter: int
    converged: bool
    family: str = "mvt_homog"


def baum_welch_homog_mvt(X, K, rng=None, max_iter=120, tol=1e-4, jitter=0.0,
                         nu_init: float = 8.0, init: Optional["HomogMVTFit"] = None):
    """EM for a K-state multivariate Student-t HMM, homogeneous transitions.

    Initialised from a Gaussian k-means seed with a moderately fat ν₀; each EM
    iteration runs the standard forward-backward with t-pdf emissions then the
    ECM mean/scale/dof update."""
    if init is not None:
        mu = init.mu.copy(); Sigma = init.Sigma.copy()
        log_pi = init.log_pi.copy(); log_A = init.log_A.copy(); nu = init.nu.copy()
    else:
        assert rng is not None, "rng required for cold-start init"
        mu, Sigma, pi = kmeans_init_mvn(X, K, rng, jitter=jitter)
        log_pi = np.log(pi + 1e-12)
        A0 = np.full((K, K), 0.05 / (K - 1)); np.fill_diagonal(A0, 0.95)
        log_A = np.log(A0 + 1e-12)
        nu = np.full(K, float(nu_init))

    prev_ll = -np.inf
    converged = False
    it = 0
    for it in range(max_iter):
        log_B = mvt_log_emissions(X, mu, Sigma, nu)
        log_gamma, log_xi, ll = forward_backward(log_pi, log_A, log_B)
        log_pi, log_A = m_step_init_trans(log_gamma, log_xi)
        mu, Sigma, nu = m_step_mvt(X, log_gamma, mu, Sigma, nu)
        if np.isfinite(ll) and abs(ll - prev_ll) < tol:
            converged = True
            it += 1
            break
        prev_ll = ll
    return HomogMVTFit(log_pi, log_A, mu, Sigma, nu, ll, it, converged)


def fit_homog_mvt_with_restarts(X, K, n_restarts=3, seed=20260603,
                                init: Optional["HomogMVTFit"] = None):
    if init is not None:
        return baum_welch_homog_mvt(X, K, init=init, max_iter=40)
    best = None
    master = np.random.default_rng(seed)
    for r in range(n_restarts):
        rng = np.random.default_rng(master.integers(1 << 31))
        jitter = 0.0 if r == 0 else 0.25
        fit = baum_welch_homog_mvt(X, K, rng=rng, jitter=jitter)
        if best is None or (np.isfinite(fit.log_likelihood) and
                            fit.log_likelihood > best.log_likelihood):
            best = fit
    if best is None:
        raise RuntimeError("all homogeneous-MVT restarts failed")
    return best


# ---------------------------------------------------------------------------
# Proposal model — MVN/MVT emissions + (NH or homogeneous) transitions, filtering
# ---------------------------------------------------------------------------

class ProposalModel:
    """K-state multivariate-Gaussian NH-HMM on (rₜ, σₜ^{5d}, dₜ_{200}).

    forward_posterior returns (T, 3) filtering posteriors in canonical
    (bear, ranging, bull) order via a μ-sort on the *return* channel (col 0):
    lowest mean return -> bear, highest -> bull. K>3 sums the extra states into
    the nearest canonical bucket via the same return-ordering rule.

    Standardisation (feature z-scoring + VIX z-scoring) uses training-window
    statistics stored at fit time, so applying the model forward stays causal.
    """

    def __init__(self, fit, feat_mean, feat_std, vix_mean, vix_std, K,
                 feature_fn=None, order_channel: int = 0):
        self.fit = fit
        self.feat_mean = feat_mean
        self.feat_std = feat_std
        self.vix_mean = vix_mean
        self.vix_std = vix_std
        self.K = K
        self.feature_fn = feature_fn if feature_fn is not None else trivariate_features
        self.is_nh = hasattr(fit, "W")                 # NH fit carries softmax W
        self.is_mvt = hasattr(fit, "nu")               # Student-t fit carries dof
        order = np.argsort(fit.mu[:, order_channel])   # by return (or drawdown) channel
        # Map each state -> canonical column. For K=3: order[0]=bear, [1]=ranging,
        # [2]=bull. For K>3, first third -> bear, last third -> bull, middle -> ranging.
        col = np.empty(K, dtype=int)
        if K == 3:
            col[order[0]] = 0
            col[order[1]] = 1
            col[order[2]] = 2
        else:
            n_bear = K // 3
            n_bull = K // 3
            for rank, st in enumerate(order):
                if rank < n_bear:
                    col[st] = 0
                elif rank >= K - n_bull:
                    col[st] = 2
                else:
                    col[st] = 1
        self.state_to_col = col

    def _standardize(self, F, vix):
        Xz = (F - self.feat_mean[None, :]) / self.feat_std[None, :]
        vz = (vix - self.vix_mean) / self.vix_std
        Xc = np.column_stack([np.ones_like(vz), vz])
        return Xz, Xc

    def forward_posterior(self, data: pd.DataFrame) -> np.ndarray:
        F, valid = self.feature_fn(data)
        vix = vix_covariate(data)
        if valid.sum() < 2:
            return np.full((len(data), 3), np.nan)

        Xz, Xc = self._standardize(F[valid], vix[valid])
        if self.is_mvt:
            log_B = mvt_log_emissions(Xz, self.fit.mu, self.fit.Sigma, self.fit.nu)
        else:
            log_B = mvn_log_emissions(Xz, self.fit.mu, self.fit.Sigma)
        if self.is_nh:
            log_A_seq = nh_log_A_seq(self.fit.W, Xc)
            log_alpha = forward_nh(self.fit.log_pi, log_A_seq, log_B)
        else:
            log_alpha = forward(self.fit.log_pi, self.fit.log_A, log_B)
        log_post = log_alpha - logsumexp(log_alpha, axis=1, keepdims=True)
        post = np.exp(log_post)                         # (T_valid, K)

        post3 = np.zeros((post.shape[0], 3))
        for st in range(self.K):
            post3[:, self.state_to_col[st]] += post[:, st]

        full = np.full((len(data), 3), np.nan)
        full[valid] = post3
        return full


def make_proposal_factory(K: int = 3, n_restarts_cold: int = 3,
                          feature_fn=None, transitions: str = "nh",
                          emission: str = "mvn", order_channel: int = 0):
    """Stateful factory for the MVN/MVT proposal family: cold-start k-means +
    jittered restarts on the first call, warm-start single-EM thereafter (the
    walk-forward tractability trick from the baseline, carried over here).

    `transitions` selects 'nh' (softmax-VIX, exp_002/exp_004) or 'homog'
    (closed-form, exp_003/exp_005). `emission` selects 'mvn' (Gaussian) or 'mvt'
    (Student-t, exp_005). `feature_fn` selects the observation channels
    (trivariate by default; drawdown-only for exp_004).

    Note: 'mvt' is only wired for homogeneous transitions (the Student-t ECM
    M-step + softmax-transition optimise are not combined in this iteration)."""
    if emission == "mvt" and transitions != "homog":
        raise NotImplementedError("mvt emission is only wired for homogeneous transitions")
    if feature_fn is None:
        feature_fn = trivariate_features
    state = {"last_fit": None}

    def factory(train_data: pd.DataFrame) -> ProposalModel:
        F, valid = feature_fn(train_data)
        vix = vix_covariate(train_data)
        Fv = F[valid]
        vixv = vix[valid]
        if len(Fv) < 50:
            raise RuntimeError(f"Not enough valid observations to fit: {len(Fv)}")

        feat_mean = Fv.mean(axis=0)
        feat_std = Fv.std(axis=0) + 1e-9
        vix_mean = float(vixv.mean())
        vix_std = float(vixv.std() + 1e-9)

        Xz = (Fv - feat_mean[None, :]) / feat_std[None, :]
        vz = (vixv - vix_mean) / vix_std
        Xc = np.column_stack([np.ones_like(vz), vz])

        if transitions == "nh":
            if state["last_fit"] is None:
                fit = fit_nh_mvn_with_restarts(Xz, Xc, K=K, n_restarts=n_restarts_cold)
            else:
                fit = fit_nh_mvn_with_restarts(Xz, Xc, K=K, init=state["last_fit"])
        elif transitions == "homog":
            if emission == "mvt":
                fit_fn = fit_homog_mvt_with_restarts
            else:
                fit_fn = fit_homog_mvn_with_restarts
            if state["last_fit"] is None:
                fit = fit_fn(Xz, K=K, n_restarts=n_restarts_cold)
            else:
                fit = fit_fn(Xz, K=K, init=state["last_fit"])
        else:
            raise ValueError(f"unknown transitions={transitions!r}")

        state["last_fit"] = fit
        return ProposalModel(fit, feat_mean, feat_std, vix_mean, vix_std, K,
                             feature_fn=feature_fn, order_channel=order_channel)

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


def run_exp_002_proposal_k3() -> None:
    """K=3 multivariate Gaussian (full Σ) on (rₜ, σₜ^{5d}, dₜ_{200}), NH-HMM
    transitions with VIX as the covariate (x_t = (1, VIX_t)).

    The literature-tier proposal: richer observation channels + regime-switching
    driven by an exogenous fear gauge.
    """
    factory = make_proposal_factory(K=3, n_restarts_cold=3)
    harness.evaluate(
        experiment_id="exp_002_proposal_k3",
        model_factory=factory,
        observation="trivar_r_vol5_dd200",
        K=3,
        emission="mvn_full",
        transitions="nh_softmax_vix",
        comment="Proposal: K=3 full-Sigma MVN on (r,vol5,dd200), NH-HMM softmax(VIX). Return-sort labelling.",
    )


def run_exp_003_multivariate_only() -> None:
    """Ablation: K=3 full-Sigma MVN on (rₜ, σₜ^{5d}, dₜ_{200}) with HOMOGENEOUS
    transitions. Isolates whether the multivariate observation alone helps,
    holding the transition structure fixed at the baseline's homogeneous form."""
    factory = make_proposal_factory(K=3, n_restarts_cold=3,
                                    feature_fn=trivariate_features,
                                    transitions="homog", order_channel=0)
    harness.evaluate(
        experiment_id="exp_003_multivariate_only",
        model_factory=factory,
        observation="trivar_r_vol5_dd200",
        K=3,
        emission="mvn_full",
        transitions="homogeneous",
        comment="Ablation: multivariate obs alone (trivar MVN, homogeneous A). Isolates the observation channel.",
    )


def run_exp_004_nh_only() -> None:
    """Ablation: K=3 Gaussian on the baseline's univariate rolling-200 drawdown,
    but with NH-HMM softmax(VIX) transitions. Isolates whether the
    non-homogeneous transitions alone help, holding the observation at the
    baseline's single drawdown channel."""
    factory = make_proposal_factory(K=3, n_restarts_cold=3,
                                    feature_fn=drawdown_only_features,
                                    transitions="nh", order_channel=0)
    harness.evaluate(
        experiment_id="exp_004_nh_only",
        model_factory=factory,
        observation="drawdown_200",
        K=3,
        emission="gaussian",
        transitions="nh_softmax_vix",
        comment="Ablation: NH transitions alone (drawdown-200 univariate, softmax(VIX) A). Isolates the transition.",
    )


def run_exp_005_mvt_homog() -> None:
    """Phase 3: K=3 multivariate Student-t (full scale Σ, per-state dof ν) on
    (rₜ, σₜ^{5d}, dₜ_{200}) with homogeneous transitions.

    Hypothesis: exp_003 (Gaussian, homogeneous, multivariate) was one period from
    a finite score — only COVID (0.000) failed, because the 28-day fat-tailed
    crash gives every Gaussian state vanishing likelihood, so the sticky filter
    never switches to bear. Fat-tailed t emissions should let the bear state
    claim the COVID extremes (and the deep 2018/2022 tails) without disturbing
    exp_003's preserved 2024 bull — no NH transition, so no calm-VIX 2024 collapse."""
    factory = make_proposal_factory(K=3, n_restarts_cold=3,
                                    feature_fn=trivariate_features,
                                    transitions="homog", emission="mvt",
                                    order_channel=0)
    harness.evaluate(
        experiment_id="exp_005_mvt_homog",
        model_factory=factory,
        observation="trivar_r_vol5_dd200",
        K=3,
        emission="mvt_full",
        transitions="homogeneous",
        comment="Phase3: K=3 multivariate Student-t (full Sigma, per-state nu), homogeneous A. Fat tails to crack COVID.",
    )


EXPERIMENTS = {
    "exp_001_baseline": run_exp_001_baseline,
    "exp_002_proposal_k3": run_exp_002_proposal_k3,
    "exp_003_multivariate_only": run_exp_003_multivariate_only,
    "exp_004_nh_only": run_exp_004_nh_only,
    "exp_005_mvt_homog": run_exp_005_mvt_homog,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=list(EXPERIMENTS.keys()))
    args = parser.parse_args()
    EXPERIMENTS[args.experiment]()


if __name__ == "__main__":
    sys.exit(main())
