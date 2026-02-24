import numpy as np

def _safe_log(x):
    return np.log(np.maximum(x, 1e-300))

def build_simple_hmm(n_states: int, diag_a: float = 0.80, diag_b: float = 0.85):
    A = np.full((n_states, n_states), (1.0 - diag_a) / max(n_states - 1, 1), dtype=float)
    np.fill_diagonal(A, diag_a)

    B = np.full((n_states, n_states), (1.0 - diag_b) / max(n_states - 1, 1), dtype=float)
    np.fill_diagonal(B, diag_b)

    pi = np.ones(n_states, dtype=float) / n_states
    return A, B, pi

def viterbi(obs_idx, A, B, pi):
    T = len(obs_idx)
    N = A.shape[0]
    dp = np.full((T, N), -np.inf, dtype=float)
    back = np.zeros((T, N), dtype=int)

    dp[0] = _safe_log(pi) + _safe_log(B[:, obs_idx[0]])

    for t in range(1, T):
        for j in range(N):
            scores = dp[t - 1] + _safe_log(A[:, j])
            back[t, j] = int(np.argmax(scores))
            dp[t, j] = float(np.max(scores)) + _safe_log(B[j, obs_idx[t]])

    last = int(np.argmax(dp[T - 1]))
    path = [last]
    for t in range(T - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    path.reverse()
    return path
