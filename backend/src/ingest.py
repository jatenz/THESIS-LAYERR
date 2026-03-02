import re
import pandas as pd

def norm_col(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("\ufeff", "")
    s = re.sub(r"[%]", "percent", s)
    s = re.sub(r"[()]", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding_errors="ignore")
    df.columns = [norm_col(c) for c in df.columns]
    return df

def detect_mode(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    if any(c in cols for c in ["date", "day", "week", "month"]):
        return "time_series"
    if "video_publish_time" in cols or "published_at" in cols:
        return "per_video"
    return "unknown"

def pick_first_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def drop_totals_like_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    if "content" in out.columns:
        m = out["content"].astype(str).str.strip().str.lower()
        out = out[m != "total"].copy()
    if "video_title" in out.columns:
        m2 = out["video_title"].astype(str).str.strip().str.lower()
        out = out[m2 != "total"].copy()
    return out
