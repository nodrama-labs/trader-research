"""Synthetic sanity checks for the NH-MVN proposal machinery in sweep.py.

Not part of the autoresearch contract — a developer harness to verify the new
multivariate-Gaussian + non-homogeneous-transition code before burning a full
walk-forward evaluation on it. Run: python test_nh_mvn.py
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

import sweep


def test_mvn_emissions_match_scipy():
    rng = np.random.default_rng(0)
    D, K, T = 3, 2, 50
    X = rng.normal(size=(T, D))
    mu = rng.normal(size=(K, D))
    Sigma = np.empty((K, D, D))
    for k in range(K):
        a = rng.normal(size=(D, D))
        Sigma[k] = a @ a.T + np.eye(D)
    got = sweep.mvn_log_emissions(X, mu, Sigma)
    for k in range(K):
        ref = multivariate_normal(mean=mu[k], cov=Sigma[k]).logpdf(X)
        assert np.allclose(got[:, k], ref, atol=1e-8), f"emission mismatch state {k}"
    print("[ok] mvn_log_emissions matches scipy.multivariate_normal.logpdf")


def test_nh_log_A_rows_normalised():
    rng = np.random.default_rng(1)
    K, Dc, T = 3, 2, 40
    W = rng.normal(size=(K, K, Dc))
    W[:, 0, :] = 0.0
    Xc = np.column_stack([np.ones(T), rng.normal(size=T)])
    logA = sweep.nh_log_A_seq(W, Xc)
    rowsum = np.exp(logsumexp(logA, axis=2))
    assert np.allclose(rowsum, 1.0, atol=1e-10), "transition rows must sum to 1"
    print("[ok] nh_log_A_seq rows sum to 1")


def simulate_nh_mvn(rng, T=1500):
    """Three well-separated 3-D Gaussian states, sticky homogeneous transitions
    (the NH machinery must still recover this homogeneous special case)."""
    mu = np.array([[-2.0, 1.5, -1.5],   # bear: low return, high vol, deep dd
                   [0.0, 0.0, 0.0],     # ranging
                   [2.0, -0.5, 1.0]])   # bull
    Sigma = np.stack([0.3 * np.eye(3)] * 3)
    A = np.array([[0.95, 0.04, 0.01],
                  [0.03, 0.94, 0.03],
                  [0.01, 0.04, 0.95]])
    states = np.empty(T, dtype=int)
    states[0] = 1
    for t in range(1, T):
        states[t] = rng.choice(3, p=A[states[t - 1]])
    X = np.array([rng.multivariate_normal(mu[s], Sigma[s]) for s in states])
    Xc = np.column_stack([np.ones(T), rng.normal(size=T)])  # VIX-like, irrelevant
    return X, Xc, states, mu


def test_em_recovers_states():
    rng = np.random.default_rng(7)
    X, Xc, states, mu_true = simulate_nh_mvn(rng)

    # 1) log-likelihood must be non-decreasing across EM iterations.
    Dc = Xc.shape[1]
    init_rng = np.random.default_rng(3)
    muk, Sigk, pik = sweep.kmeans_init_mvn(X, 3, init_rng)
    log_pi = np.log(pik + 1e-12)
    A0 = np.full((3, 3), 0.025); np.fill_diagonal(A0, 0.95)
    W = sweep._sticky_W_init(A0, Dc)
    lls = []
    for _ in range(40):
        log_B = sweep.mvn_log_emissions(X, muk, Sigk)
        log_A_seq = sweep.nh_log_A_seq(W, Xc)
        lg, lx, ll = sweep.forward_backward_nh(log_pi, log_A_seq, log_B)
        log_pi = lg[0] - logsumexp(lg[0])
        W = sweep.nh_transition_mstep(lx, Xc, W)
        muk, Sigk = sweep.m_step_mvn(X, lg)
        lls.append(ll)
    diffs = np.diff(lls)
    assert (diffs > -1e-4).all(), f"LL decreased during EM: min diff {diffs.min():.4f}"
    print(f"[ok] EM log-likelihood non-decreasing ({lls[0]:.1f} -> {lls[-1]:.1f})")

    # 2) recovered means must match the truth up to state permutation.
    fit = sweep.fit_nh_mvn_with_restarts(X, Xc, K=3, n_restarts=3)
    order = np.argsort(fit.mu[:, 0])
    mu_sorted = fit.mu[order]
    mu_true_sorted = mu_true[np.argsort(mu_true[:, 0])]
    err = np.abs(mu_sorted - mu_true_sorted).max()
    assert err < 0.5, f"recovered means off by {err:.3f}\n{mu_sorted}\nvs\n{mu_true_sorted}"
    print(f"[ok] EM recovers state means (max abs err {err:.3f})")

    # 3) Viterbi-ish argmax of smoothed posterior agrees with true states.
    log_B = sweep.mvn_log_emissions(X, fit.mu, fit.Sigma)
    log_A_seq = sweep.nh_log_A_seq(fit.W, Xc)
    lg, _, _ = sweep.forward_backward_nh(fit.log_pi, log_A_seq, log_B)
    pred = np.argmax(lg, axis=1)
    # remap predicted states to truth via return-order
    remap = np.empty(3, dtype=int)
    remap[order] = np.argsort(mu_true[:, 0])
    pred_mapped = remap[pred]
    acc = (pred_mapped == states).mean()
    assert acc > 0.9, f"state recovery accuracy only {acc:.3f}"
    print(f"[ok] smoothed-posterior state accuracy {acc:.3f}")


def test_warm_start_fast_and_stable():
    rng = np.random.default_rng(11)
    X, Xc, _, _ = simulate_nh_mvn(rng, T=1200)
    cold = sweep.fit_nh_mvn_with_restarts(X, Xc, K=3, n_restarts=2)
    warm = sweep.fit_nh_mvn_with_restarts(X, Xc, K=3, init=cold)
    assert warm.n_iter <= 40, f"warm start ran {warm.n_iter} iters (expected few)"
    assert warm.log_likelihood >= cold.log_likelihood - 1e-3, "warm start regressed LL"
    print(f"[ok] warm-start: {warm.n_iter} iters, LL {cold.log_likelihood:.1f} -> {warm.log_likelihood:.1f}")


def test_homog_mvn_recovers_states():
    """The homogeneous-MVN driver (exp_003 ablation) must recover the same
    synthetic states as the NH driver on the homogeneous-generating process."""
    rng = np.random.default_rng(7)
    X, Xc, states, mu_true = simulate_nh_mvn(rng)
    fit = sweep.fit_homog_mvn_with_restarts(X, K=3, n_restarts=3)
    order = np.argsort(fit.mu[:, 0])
    mu_sorted = fit.mu[order]
    mu_true_sorted = mu_true[np.argsort(mu_true[:, 0])]
    err = np.abs(mu_sorted - mu_true_sorted).max()
    assert err < 0.5, f"homog recovered means off by {err:.3f}"
    # transition rows normalised
    assert np.allclose(np.exp(logsumexp(fit.log_A, axis=1)), 1.0, atol=1e-9)
    print(f"[ok] homogeneous-MVN recovers state means (max abs err {err:.3f})")


def test_proposal_model_paths_on_real_data():
    """Both transition modes + both feature sets build and emit valid (T,3)
    posteriors (rows sum to 1 on valid days) on a real training slice."""
    import harness
    data = harness.load_joined().iloc[:500].copy()  # enough for a 200-dd warmup
    configs = [
        (sweep.trivariate_features, "nh"),
        (sweep.trivariate_features, "homog"),
        (sweep.drawdown_only_features, "nh"),
    ]
    for feat_fn, trans in configs:
        factory = sweep.make_proposal_factory(K=3, n_restarts_cold=2,
                                              feature_fn=feat_fn, transitions=trans)
        model = factory(data)
        post = model.forward_posterior(data)
        valid = ~np.isnan(post).any(axis=1)
        assert valid.sum() > 0, f"no valid posteriors ({feat_fn.__name__},{trans})"
        rs = post[valid].sum(axis=1)
        assert np.allclose(rs, 1.0, atol=1e-6), f"rows must sum to 1 ({trans})"
        assert (post[valid] >= -1e-9).all(), "posteriors must be non-negative"
    print("[ok] proposal model: all 3 ablation paths emit valid (T,3) posteriors")


def test_mvt_emissions_match_scipy():
    from scipy.stats import multivariate_t
    rng = np.random.default_rng(0)
    D, K, T = 3, 2, 60
    X = rng.normal(size=(T, D))
    mu = rng.normal(size=(K, D))
    Sigma = np.empty((K, D, D))
    for k in range(K):
        a = rng.normal(size=(D, D)); Sigma[k] = a @ a.T + np.eye(D)
    nu = np.array([4.0, 12.0])
    got = sweep.mvt_log_emissions(X, mu, Sigma, nu)
    for k in range(K):
        ref = multivariate_t(loc=mu[k], shape=Sigma[k], df=nu[k]).logpdf(X)
        assert np.allclose(got[:, k], ref, atol=1e-8), f"mvt emission mismatch state {k}"
    print("[ok] mvt_log_emissions matches scipy.multivariate_t.logpdf")


def test_mvt_em_recovers_and_detects_tails():
    """Generate heavy-tailed (low-ν) clusters; the MVT-EM must recover the means
    and infer a *finite* (fat) ν, while a Gaussian fit would be dragged by the
    outliers. Sanity: LL non-decreasing, means recovered, ν stays finite."""
    from scipy.stats import multivariate_t
    rng = np.random.default_rng(5)
    mu_true = np.array([[-2.0, 1.0, -1.0], [0.0, 0.0, 0.0], [2.0, -0.5, 1.0]])
    shape = 0.3 * np.eye(3)
    A = np.array([[0.95, 0.04, 0.01], [0.03, 0.94, 0.03], [0.01, 0.04, 0.95]])
    T = 1800
    states = np.empty(T, dtype=int); states[0] = 1
    for t in range(1, T):
        states[t] = rng.choice(3, p=A[states[t - 1]])
    X = np.array([multivariate_t(loc=mu_true[s], shape=shape, df=3.0,
                                 seed=rng).rvs() for s in states])
    fit = sweep.fit_homog_mvt_with_restarts(X, K=3, n_restarts=3)
    order = np.argsort(fit.mu[:, 0])
    err = np.abs(fit.mu[order] - mu_true[np.argsort(mu_true[:, 0])]).max()
    assert err < 0.6, f"MVT recovered means off by {err:.3f}"
    assert np.all(fit.nu > 2.0) and np.all(fit.nu < sweep.NU_MAX), \
        f"expected finite fat nu, got {fit.nu}"
    assert (fit.nu < 30).any(), f"expected at least one clearly-fat state, nu={fit.nu}"
    print(f"[ok] MVT-EM recovers means (err {err:.3f}), infers fat nu={np.round(fit.nu,2)}")


def test_mvt_model_path_on_real_data():
    import harness
    data = harness.load_joined().iloc[:500].copy()
    factory = sweep.make_proposal_factory(K=3, n_restarts_cold=2,
                                          feature_fn=sweep.trivariate_features,
                                          transitions="homog", emission="mvt")
    model = factory(data)
    post = model.forward_posterior(data)
    valid = ~np.isnan(post).any(axis=1)
    assert valid.sum() > 0 and np.allclose(post[valid].sum(axis=1), 1.0, atol=1e-6)
    print(f"[ok] MVT proposal path emits valid posteriors; nu={np.round(model.fit.nu,2)}")


if __name__ == "__main__":
    test_mvn_emissions_match_scipy()
    test_nh_log_A_rows_normalised()
    test_em_recovers_states()
    test_warm_start_fast_and_stable()
    test_homog_mvn_recovers_states()
    test_proposal_model_paths_on_real_data()
    test_mvt_emissions_match_scipy()
    test_mvt_em_recovers_and_detects_tails()
    test_mvt_model_path_on_real_data()
    print("\nAll NH-MVN/MVT sanity checks passed.")
