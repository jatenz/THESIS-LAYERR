import csv
import os

def load_channel_observations(csv_filename, target_channel_id=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "data", csv_filename)

    observations = []
    channel_name = None

    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            if target_channel_id and row["channel_id"] != target_channel_id:
                continue

            if channel_name is None:
                channel_name = row.get("channel_title", "UNKNOWN")

            observations.append(row["observation"])

    if not observations:
        raise ValueError("No rows found for that channel_id")

    return channel_name, observations
