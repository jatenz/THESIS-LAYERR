import pandas as pd

def make_summary(state_order, steady_probs, detected_mode, aggregated_rows, alpha, churn_k, k_steps):
    row = {
        "detected_mode": detected_mode,
        "aggregated_rows": int(aggregated_rows),
        "laplace_alpha": float(alpha),
        "churn_inactive_streak_k": int(churn_k) if churn_k is not None else "",
        "forecast_steps": ",".join([str(int(k)) for k in k_steps]),
    }
    for s, p in zip(state_order, steady_probs):
        row[f"steady_{s}"] = float(p)
    return pd.DataFrame([row])
