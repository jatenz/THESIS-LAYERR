import numpy as np
import pandas as pd

def transition_counts(states, order):
    idx = {s: i for i, s in enumerate(order)}
    n = len(order)
    C = np.zeros((n, n), dtype=float)
    for a, b in zip(states[:-1], states[1:]):
        if a in idx and b in idx:
            C[idx[a], idx[b]] += 1.0
    return C

def transition_matrix_smoothed(C, alpha: float = 1.0):
    C = np.asarray(C, dtype=float)
    C2 = C + float(alpha)
    rs = C2.sum(axis=1, keepdims=True)
    P = np.divide(C2, rs, out=np.zeros_like(C2), where=rs != 0)
    return P

def enforce_absorbing(P, order, absorbing_state="CHURN"):
    P = np.asarray(P, dtype=float).copy()
    if absorbing_state in order:
        i = order.index(absorbing_state)
        P[i, :] = 0.0
        P[i, i] = 1.0
    return P

def steady_state(P, tol=1e-12, iters=200000):
    n = P.shape[0]
    v = np.ones(n) / n
    for _ in range(iters):
        v2 = v @ P
        if np.linalg.norm(v2 - v, 1) < tol:
            return v2
        v = v2
    return v

def k_step_forecast(P, state_order, start_state, k_list):
    idx = {s:i for i,s in enumerate(state_order)}
    if start_state not in idx:
        raise ValueError("start_state not in state_order")
    v = np.zeros((len(state_order),), dtype=float)
    v[idx[start_state]] = 1.0

    out_rows = []
    for k in k_list:
        Pk = np.linalg.matrix_power(P, int(k))
        dist = v @ Pk
        row = {"k": int(k), "start_state": start_state}
        for s, p in zip(state_order, dist.tolist()):
            row[f"prob_{s}"] = float(p)
        out_rows.append(row)
    return pd.DataFrame(out_rows)

def expected_time_to_absorption(P, order, absorbing_state="CHURN"):
    if absorbing_state not in order:
        raise ValueError("absorbing_state not in order")

    P = np.asarray(P, dtype=float)
    a = order.index(absorbing_state)
    transient = [i for i in range(len(order)) if i != a]
    Q = P[np.ix_(transient, transient)]

    I = np.eye(Q.shape[0], dtype=float)
    try:
        N = np.linalg.inv(I - Q)
    except np.linalg.LinAlgError:
        N = np.linalg.pinv(I - Q)

    ones = np.ones((Q.shape[0], 1), dtype=float)
    t = N @ ones

    rows = []
    for i, ti in enumerate(t.flatten().tolist()):
        rows.append({"state": order[transient[i]], "expected_steps_to_churn": float(ti)})
    rows.append({"state": absorbing_state, "expected_steps_to_churn": 0.0})
    return pd.DataFrame(rows), N

def frames(C, P, order):
    return (pd.DataFrame(C, index=order, columns=order),
            pd.DataFrame(P, index=order, columns=order))
