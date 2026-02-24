import pandas as pd

BASE_STATES = ["NEW", "ENGAGED", "LOYAL", "INACTIVE"]

def classify_states_4(g: pd.DataFrame) -> pd.DataFrame:
    out = g.copy()
    score = out["loyalty_score"]
    q25 = float(score.quantile(0.25))
    q60 = float(score.quantile(0.60))
    q85 = float(score.quantile(0.85))

    states = []
    for v in score.tolist():
        if v <= q25:
            states.append("INACTIVE")
        elif v <= q60:
            states.append("NEW")
        elif v <= q85:
            states.append("ENGAGED")
        else:
            states.append("LOYAL")
    out["state_obs"] = states
    return out

def add_absorbing_churn(g: pd.DataFrame, inactive_streak_k: int = 3) -> pd.DataFrame:
    out = g.copy()
    if "state_obs" not in out.columns:
        raise ValueError("state_obs column missing. Call classify_states_4 first.")

    churn_states = []
    streak = 0
    for s in out["state_obs"].astype(str).tolist():
        if s == "INACTIVE":
            streak += 1
        else:
            streak = 0
        churn_states.append("CHURN" if streak >= inactive_streak_k else s)
    out["state_obs"] = churn_states
    return out
