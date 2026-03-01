import pandas as pd

def to_number_series(x):
    if isinstance(x, pd.Series):
        s = x.astype(str)
    else:
        s = pd.Series(x).astype(str)
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace("—", "", regex=False)
    s = s.str.replace("–", "", regex=False)
    s = s.replace({"nan": "0", "None": "0", "": "0"})
    return pd.to_numeric(s, errors="coerce").fillna(0)

def parse_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "watch_time_hours": "watch_time_hours",
        "watch_time_minutes": "watch_time_minutes",
        "watch_time_min": "watch_time_minutes",
        "watch_time": "watch_time_minutes",
        "watch_timehours": "watch_time_hours",
        "views": "views",
        "subscribers": "subscribers",
        "likes": "likes",
        "comments": "comments",
        "shares": "shares",
        "returning_viewers": "returning_viewers",
        "impressions": "impressions",
        "impressions_click_through_rate_percent": "impressions_ctr_percent",
        "impressions_click_through_rate": "impressions_ctr_percent",
        "impressions_ctr_percent": "impressions_ctr_percent",
    }
    return df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

def ensure_metrics(df: pd.DataFrame) -> pd.DataFrame:
    cols = set(df.columns)
    metrics = [
        "views",
        "watch_time_hours",
        "watch_time_minutes",
        "subscribers",
        "likes",
        "comments",
        "shares",
        "returning_viewers",
        "impressions",
    ]
    out = df.copy()
    for m in metrics:
        if m not in cols:
            out[m] = 0

    if "watch_time_hours" in out.columns and "watch_time_minutes" in out.columns:
        h = to_number_series(out["watch_time_hours"])
        mi = to_number_series(out["watch_time_minutes"])
        if h.sum() > 0 and mi.sum() == 0:
            out["watch_time_minutes"] = h * 60.0
        elif mi.sum() > 0 and h.sum() == 0:
            out["watch_time_hours"] = mi / 60.0
    return out

def build_time_series_from_time_series_export(df: pd.DataFrame, date_col: str, freq: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = parse_date_series(out[date_col])
    out = out.dropna(subset=[date_col]).sort_values(date_col)
    out = ensure_metrics(out)
    for m in ["views","watch_time_minutes","subscribers","likes","comments","shares","returning_viewers","impressions"]:
        out[m] = to_number_series(out[m])
    out = out.rename(columns={date_col: "period"})
    numeric = ["views","watch_time_minutes","subscribers","likes","comments","shares","returning_viewers","impressions"]
    g = out.groupby(pd.Grouper(key="period", freq=freq))[numeric].sum().reset_index()
    return g.sort_values("period").reset_index(drop=True)

def build_time_series_from_per_video_export(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    out = ensure_metrics(df.copy())
    date_col = "video_publish_time" if "video_publish_time" in out.columns else ("published_at" if "published_at" in out.columns else None)
    if date_col is None:
        raise ValueError("Per-video export detected but no publish date column found.")
    out[date_col] = parse_date_series(out[date_col])
    out = out.dropna(subset=[date_col]).sort_values(date_col)
    for m in ["views","watch_time_minutes","subscribers","likes","comments","shares","returning_viewers","impressions"]:
        out[m] = to_number_series(out[m])
    out = out.rename(columns={date_col: "period"})
    numeric = ["views","watch_time_minutes","subscribers","likes","comments","shares","returning_viewers","impressions"]
    g = out.groupby(pd.Grouper(key="period", freq=freq))[numeric].sum().reset_index()
    return g.sort_values("period").reset_index(drop=True)

def zscore_safe(s: pd.Series) -> pd.Series:
    s = to_number_series(s)
    mu = float(s.mean())
    sd = float(s.std(ddof=0))
    if sd == 0:
        return pd.Series([0.0] * len(s))
    return (s - mu) / sd

def add_features(g: pd.DataFrame, w_watch=0.40, w_return=0.30, w_views=0.20, w_engage=0.10) -> pd.DataFrame:
    out = ensure_metrics(g.copy())
    out["engagement_actions"] = to_number_series(out["likes"]) + to_number_series(out["comments"]) + to_number_series(out["shares"])
    out["views_z"] = zscore_safe(out["views"])
    out["watch_z"] = zscore_safe(out["watch_time_minutes"])
    out["return_z"] = zscore_safe(out["returning_viewers"])
    out["engage_z"] = zscore_safe(out["engagement_actions"])
    out["loyalty_score"] = w_watch*out["watch_z"] + w_return*out["return_z"] + w_views*out["views_z"] + w_engage*out["engage_z"]
    return out
