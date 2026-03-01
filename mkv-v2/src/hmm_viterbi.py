import numpy as np

def _safe_log(x):
    return np.log(np.maximum(x, 1e-300))

def _row_norm(M):
    M = np.asarray(M, dtype=float)
    rs = M.sum(axis=1, keepdims=True)
    return np.divide(M, rs, out=np.zeros_like(M), where=rs != 0)

def build_hmm_dirichlet(n_states: int, seed: int = 7):
    rng = np.random.default_rng(int(seed))
    A = rng.random((n_states, n_states)) + 1e-3
    B = rng.random((n_states, n_states)) + 1e-3
    pi = rng.random((n_states,)) + 1e-3
    A = _row_norm(A)
    B = _row_norm(B)
    pi = pi / pi.sum()
    return A, B, pi

def forward_backward_scaled(obs_idx, A, B, pi):
    obs = np.asarray(obs_idx, dtype=int)
    T = len(obs)
    N = A.shape[0]

    alpha = np.zeros((T, N), dtype=float)
    c = np.zeros((T,), dtype=float)

    alpha[0] = pi * B[:, obs[0]]
    c[0] = alpha[0].sum()
    if c[0] == 0:
        c[0] = 1e-300
    alpha[0] /= c[0]

    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ A) * B[:, obs[t]]
        c[t] = alpha[t].sum()
        if c[t] == 0:
            c[t] = 1e-300
        alpha[t] /= c[t]

    beta = np.zeros((T, N), dtype=float)
    beta[T - 1] = 1.0

    for t in range(T - 2, -1, -1):
        beta[t] = (A @ (B[:, obs[t + 1]] * beta[t + 1]))
        beta[t] /= c[t + 1]

    ll = float(np.sum(np.log(np.maximum(c, 1e-300))))
    gamma = alpha * beta
    gamma = np.divide(gamma, gamma.sum(axis=1, keepdims=True), out=np.zeros_like(gamma), where=gamma.sum(axis=1, keepdims=True) != 0)

    xi = np.zeros((T - 1, N, N), dtype=float)
    for t in range(T - 1):
        denom = (alpha[t][:, None] * A) * (B[:, obs[t + 1]] * beta[t + 1])[None, :]
        s = denom.sum()
        if s == 0:
            continue
        xi[t] = denom / s

    return ll, gamma, xi

def baum_welch(obs_idx, n_states: int, n_iter: int = 25, tol: float = 1e-6, seed: int = 7):
    obs = np.asarray(obs_idx, dtype=int)
    if len(obs) < 2:
        raise ValueError("Need at least 2 observations for HMM training.")
    if int(n_states) <= 1:
        raise ValueError("n_states must be >= 2")

    A, B, pi = build_hmm_dirichlet(int(n_states), seed=int(seed))
    prev_ll = None

    for _ in range(int(max(1, n_iter))):
        ll, gamma, xi = forward_backward_scaled(obs, A, B, pi)

        pi = gamma[0].copy()

        A_num = xi.sum(axis=0)
        A_den = gamma[:-1].sum(axis=0)[:, None]
        A = np.divide(A_num, A_den, out=np.zeros_like(A_num), where=A_den != 0)

        B_num = np.zeros_like(B)
        for t, o in enumerate(obs):
            B_num[:, o] += gamma[t]
        B_den = gamma.sum(axis=0)[:, None]
        B = np.divide(B_num, B_den, out=np.zeros_like(B_num), where=B_den != 0)

        A = _row_norm(A + 1e-12)
        B = _row_norm(B + 1e-12)
        pi = pi / np.maximum(pi.sum(), 1e-300)

        if prev_ll is not None and abs(ll - prev_ll) < float(tol):
            prev_ll = ll
            break
        prev_ll = ll

    return A, B, pi, float(prev_ll if prev_ll is not None else 0.0)

def viterbi(obs_idx, A, B, pi):
    obs_idx = list(obs_idx)
    T = len(obs_idx)
    N = A.shape[0]
    dp = np.full((T, N), -np.inf, dtype=float)
    back = np.zeros((T, N), dtype=int)

    dp[0] = _safe_log(pi) + _safe_log(B[:, obs_idx[0]])

    for t in range(1, T):
        for j in range(N):
            scores = dp[t - 1] + _safe_log(A[:, j])
            bj = int(np.argmax(scores))
            back[t, j] = bj
            dp[t, j] = float(scores[bj]) + _safe_log(B[j, obs_idx[t]])

    last = int(np.argmax(dp[T - 1]))
    path = [last]
    for t in range(T - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    path.reverse()
    return path
