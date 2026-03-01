import os
import argparse
import pandas as pd

def categorize_daily_views(dv):
    if dv >= 50000:
        return "High"
    if dv >= 10000:
        return "Medium"
    if dv > 0:
        return "Low"
    return "None"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", default=os.path.join("data", "channel_snapshots.csv"))
    ap.add_argument("--out", default=os.path.join("data", "channel_observations.csv"))
    ap.add_argument("--channel_id", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.snapshots)
    if args.channel_id:
        df = df[df["channel_id"] == args.channel_id]

    df["snapshot_utc"] = pd.to_datetime(df["snapshot_utc"], utc=True)
    df = df.sort_values(["channel_id", "snapshot_utc"])

    df["date"] = df["snapshot_utc"].dt.date.astype(str)

    df["daily_views"] = df.groupby("channel_id")["view_count"].diff().fillna(0)
    df["daily_subscribers"] = df.groupby("channel_id")["subscriber_count"].diff()

    df["observation"] = df["daily_views"].apply(categorize_daily_views)

    out_cols = ["channel_id", "channel_title", "date", "view_count", "subscriber_count", "daily_views", "daily_subscribers", "observation"]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df[out_cols].to_csv(args.out, index=False)
    print("Wrote", args.out)

if __name__ == "__main__":
    main()
