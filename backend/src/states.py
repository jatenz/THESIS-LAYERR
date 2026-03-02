import pandas as pd

BASE_STATES_4 = ["NEW", "ENGAGED", "LOYAL", "INACTIVE"]

def _ensure_cols(out: pd.DataFrame) -> pd.DataFrame:
    for c in ["views", "watch_time_minutes", "likes", "comments", "shares"]:
        if c not in out.columns:
            out[c] = 0
    if "engagement_actions" not in out.columns:
        likes = out["likes"] if "likes" in out.columns else 0
        comments = out["comments"] if "comments" in out.columns else 0
        shares = out["shares"] if "shares" in out.columns else 0
        out["engagement_actions"] = pd.to_numeric(likes, errors="coerce").fillna(0) + pd.to_numeric(comments, errors="coerce").fillna(0) + pd.to_numeric(shares, errors="coerce").fillna(0)
    return out

def _inactive_mask(out: pd.DataFrame) -> pd.Series:
    out = _ensure_cols(out)
    views = pd.to_numeric(out["views"], errors="coerce").fillna(0)
    watch = pd.to_numeric(out["watch_time_minutes"], errors="coerce").fillna(0)
    actions = pd.to_numeric(out["engagement_actions"], errors="coerce").fillna(0)
    return (views <= 0) & (watch <= 0) & (actions <= 0)

def classify_states_4(
    g: pd.DataFrame,
    new_periods: int = 2,
    engaged_thr: float = 0.25,
    loyal_thr: float = 1.00,
) -> pd.DataFrame:
    out = g.copy()
    if "loyalty_score" not in out.columns:
        raise ValueError("loyalty_score missing. Call add_features() before classify_states_4().")

    out = _ensure_cols(out)
    inactive = _inactive_mask(out)
    score = pd.to_numeric(out["loyalty_score"], errors="coerce").fillna(0.0)

    states = []
    for i, v in enumerate(score.tolist()):
        if bool(inactive.iloc[i]):
            states.append("INACTIVE")
            continue
        if int(new_periods) > 0 and i < int(new_periods):
            states.append("NEW")
            continue
        if v >= float(loyal_thr):
            states.append("LOYAL")
        elif v >= float(engaged_thr):
            states.append("ENGAGED")
        else:
            states.append("NEW")
    out["state_obs"] = states
    return out

def add_absorbing_churn(g: pd.DataFrame, inactive_streak_k: int = 3) -> pd.DataFrame:
    out = g.copy()
    if "state_obs" not in out.columns:
        raise ValueError("state_obs column missing. Call classify_states_* first.")

    k = int(max(1, inactive_streak_k))
    churn_states = []
    streak = 0
    for s in out["state_obs"].astype(str).tolist():
        if s == "INACTIVE":
            streak += 1
        else:
            streak = 0
        churn_states.append("CHURN" if streak >= k else s)
    out["state_obs"] = churn_states
    return out
