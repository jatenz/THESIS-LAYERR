import sys
import os
import argparse
from googleapiclient.discovery import build

sys.path.append(os.path.dirname(__file__))

from preprocessing import load_channel_observations
from hmm_model import states, obs_idx, A, B, pi
from viterbi import viterbi


def resolve_channel_id(youtube, name):
    if name.startswith("@"):
        r = youtube.channels().list(part="id", forHandle=name[1:]).execute()
        return r["items"][0]["id"]

    if name.startswith("UC"):
        return name

    r = youtube.search().list(
        part="snippet",
        q=name,
        type="channel",
        maxResults=1
    ).execute()

    return r["items"][0]["snippet"]["channelId"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", required=True)
    ap.add_argument("--api_key", required=True)
    args = ap.parse_args()

    youtube = build("youtube", "v3", developerKey=args.api_key)
    channel_id = resolve_channel_id(youtube, args.channel)

    channel_name, observations = load_channel_observations(
        "channel_observations.csv",
        target_channel_id=channel_id
    )

    obs_seq = [obs_idx[o] for o in observations]

    state_path_idx = viterbi(obs_seq, A, B, pi)
    state_path = [states[i] for i in state_path_idx]

    print("Channel:", channel_name)
    print("Channel ID:", channel_id)
    print("Observations:", observations)
    print("Predicted states:", state_path)

    if state_path[-1] == "Churned":
        print("Churn prediction: USER CHURNED")
    elif "AtRisk" in state_path[-2:]:
        print("Churn prediction: HIGH RISK")
    else:
        print("Churn prediction: LOW RISK")


if __name__ == "__main__":
    main()
