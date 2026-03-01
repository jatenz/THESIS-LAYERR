import time
import os
import pandas as pd
from datetime import datetime, timezone
from googleapiclient.discovery import build


API_KEY = "AIzaSyD5qvw8n8smVA6iJ8TySqy5g7z1s_cywpk" # set via environment variable
CHANNEL_NAME = "@MrBeast"           # change if needed
INTERVAL_SECONDS = 10             # 1 hour


def resolve_channel_id(youtube, name):

    if name.startswith("@"):
        r = youtube.channels().list(
            part="id",
            forHandle=name[1:]
        ).execute()
        items = r.get("items", [])
        if items:
            return items[0]["id"]

    if name.startswith("UC"):
        return name

    r = youtube.search().list(
        part="snippet",
        q=name,
        type="channel",
        maxResults=1
    ).execute()

    return r["items"][0]["snippet"]["channelId"]


def fetch_stats(youtube, cid):
    r = youtube.channels().list(
        part="snippet,statistics",
        id=cid
    ).execute()["items"][0]

    s = r["statistics"]
    sn = r["snippet"]

    return {
        "channel_id": cid,
        "channel_title": sn["title"],
        "snapshot_utc": datetime.now(timezone.utc).isoformat(),
        "view_count": int(s.get("viewCount", 0)),
        "subscriber_count": int(s.get("subscriberCount", 0)),
        "video_count": int(s.get("videoCount", 0))
    }


def append_row(row, path):
    os.makedirs("data", exist_ok=True)

    df_new = pd.DataFrame([row])

    if os.path.exists(path):
        df_old = pd.read_csv(path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(path, index=False)


def main():

    youtube = build("youtube", "v3", developerKey=API_KEY)
    cid = resolve_channel_id(youtube, CHANNEL_NAME)

    out = f"data/{cid}_snapshots.csv"

    print("Tracking:", CHANNEL_NAME, cid)
    print("Saving to:", out)
    print("Interval seconds:", INTERVAL_SECONDS)

    while True:
        try:
            row = fetch_stats(youtube, cid)
            append_row(row, out)
            print("Saved snapshot:", row["snapshot_utc"], row["view_count"])
            time.sleep(INTERVAL_SECONDS)

        except KeyboardInterrupt:
            print("Stopped by user")
            break

        except Exception as e:
            print("Error:", e)
            time.sleep(60)


if __name__ == "__main__":
    main()
