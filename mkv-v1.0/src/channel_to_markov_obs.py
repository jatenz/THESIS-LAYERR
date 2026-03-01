import pandas as pd

def categorize(daily_views):
    if daily_views >= 50000:
        return "High"
    if daily_views >= 10000:
        return "Medium"
    if daily_views > 0:
        return "Low"
    return "None"

df = pd.read_csv("data/channel_snapshots.csv")

df["snapshot_utc"] = pd.to_datetime(df["snapshot_utc"])
df = df.sort_values("snapshot_utc")

df["daily_views"] = df["view_count"].diff().fillna(0)
df["observation"] = df["daily_views"].apply(categorize)

df[["snapshot_utc","daily_views","observation"]].to_csv(
    "data/channel_markov_obs.csv",
    index=False
)

print(df[["snapshot_utc","daily_views","observation"]])
