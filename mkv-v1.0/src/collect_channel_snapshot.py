import os
import argparse
from datetime import datetime, timezone
import pandas as pd
from googleapiclient.discovery import build

def get_channel_id(youtube, channel):
    if channel.startswith("@"):
        resp = youtube.channels().list(part="id", forHandle=channel[1:]).execute()
        items = resp.get("items", [])
        if not items:
            raise ValueError("Handle not found")
        return items[0]["id"]
    if channel.startswith("UC"):
        return channel
    resp = youtube.search().list(part="snippet", q=channel, type="channel", maxResults=1).execute()
    items = resp.get("items", [])
    if not items:
        raise ValueError("Channel not found")
    return items[0]["snippet"]["channelId"]

def fetch_channel_stats(youtube, channel_id):
    resp = youtube.channels().list(part="snippet,statistics", id=channel_id).execute()
    items = resp.get("items", [])
    if not items:
        raise ValueError("Channel ID not found")
    ch = items[0]
    s = ch.get("statistics", {})
    sn = ch.get("snippet", {})
    return {
        "channel_id": channel_id,
        "channel_title": sn.get("title", ""),
        "snapshot_utc": datetime.now(timezone.utc).isoformat(),
        "subscriber_count": int(s.get("subscriberCount", 0)) if s.get("subscriberCount") is not None else None,
        "view_count": int(s.get("viewCount", 0)) if s.get("viewCount") is not None else None,
        "video_count": int(s.get("videoCount", 0)) if s.get("videoCount") is not None else None
    }

def append_csv(row, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_new = pd.DataFrame([row])
    if os.path.exists(out_csv):
        df_old = pd.read_csv(out_csv)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(out_csv, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_key", required=True)
    ap.add_argument("--channel", required=True, help="channelId (UC...), @handle, or channel name")
    ap.add_argument("--out", default=os.path.join("data", "channel_snapshots.csv"))
    args = ap.parse_args()

    youtube = build("youtube", "v3", developerKey=args.api_key)
    channel_id = get_channel_id(youtube, args.channel)
    row = fetch_channel_stats(youtube, channel_id)
    append_csv(row, args.out)
    print("Saved snapshot:", row["channel_title"], row["channel_id"], row["snapshot_utc"])

if __name__ == "__main__":
    main()
