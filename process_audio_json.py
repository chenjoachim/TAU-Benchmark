import csv
import argparse
import json
import os
import sys
import uuid
import pandas as pd
from tqdm import tqdm

from utils import (
    download_from_curl,
    download_from_google_drive,
    download_from_yt,
    crop_audio,
)

MAX_AUDIO_LENGTH = 30  # seconds
SUBSETS = [
    "Transit",
    "Retail",
    "Daily",
    "Cultural",
    "Public",
    "Emergency",
    "Broadcast",
    "Music",
    "Banking",
    "Celebrity",
]


def timestamp_to_ms(timestamp):
    """Convert a timestamp in the format 'HH:MM:SS' to milliseconds."""
    if timestamp is None or pd.isna(timestamp) or timestamp.strip() == "":
        return -1
    parts = list(map(int, timestamp.split(":")))
    if len(parts) == 2:  # MM:SS format
        minutes, seconds = parts
        hours = 0
    elif len(parts) == 3:  # HH:MM:SS format
        hours, minutes, seconds = parts
    else:
        print(f"Invalid timestamp format: {timestamp}", file=sys.stderr)
        return -1

    return (hours * 3600 + minutes * 60 + seconds) * 1000


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        required=True,
        help="Path to the input CSV file containing audio file paths.",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        required=True,
        help="Path to the output JSONL file where processed audio data will be saved.",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        choices=["mp3", "wav"],
        default="mp3",
        help="Format of the audio files to be processed (default: mp3).",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="data/audio",
        help="Directory where audio files are stored (default: data/audio).",
    )
    return parser.parse_args()


def process_audio_download(row, args, audio_idx=1):
    """Helper function to download audio for a single row."""
    try:
        if "drive" in row[f"link_{audio_idx}"]:
            return download_from_google_drive(
                row[f"link_{audio_idx}"],
                format=args.format,
                output_path=args.audio_dir,
                output_id=row["unique_id"],
                audio_idx=audio_idx,
            )
        elif "youtu" in row[f"link_{audio_idx}"]:
            return download_from_yt(
                row[f"link_{audio_idx}"].split("&")[
                    0
                ],  # Remove any additional parameters
                format=args.format,
                output_path=args.audio_dir,
                output_id=row["unique_id"],
                audio_idx=audio_idx,
            )
        else:
            return download_from_curl(
                row[f"link_{audio_idx}"],
                format=args.format,
                output_path=args.audio_dir,
                output_id=row["unique_id"],
                audio_idx=audio_idx,
            )
    except Exception as e:
        print(
            f"Error downloading audio for {row['unique_id']} (link {audio_idx}): {e}",
            file=sys.stderr,
        )
        return None


def main(args):

    os.makedirs(args.audio_dir, exist_ok=True)

    with open(args.input_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = list(reader)

    for idx, row in enumerate(tqdm(data, desc="Processing Rows")):
        row["unique_id"] = f"{idx:0>8d}"  # Generate unique ID
        audio_data = []
        for audio_idx in range(1, 4):
            if not row.get(f"link_{audio_idx}"):
                continue
            audio_path = process_audio_download(row, args)
            start_ms = timestamp_to_ms(row.get(f"start_{audio_idx}"))
            end_ms = timestamp_to_ms(row.get(f"end_{audio_idx}"))
            if audio_path:
                audio_path, start_ms, end_ms = crop_audio(
                    audio_path,
                    start_ms,
                    end_ms,
                    args.format,
                    max_length=MAX_AUDIO_LENGTH * 1000,
                )
                audio_data.append(
                    {
                        "link": row[f"link_{audio_idx}"],
                        "audio_path": audio_path,
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                    }
                )
        row["audio"] = audio_data
        for key in [
            "link_1",
            "link_2",
            "link_3",
            "start_1",
            "start_2",
            "start_3",
            "end_1",
            "end_2",
            "end_3",
        ]:
            row.pop(key, None)
        with open(args.output_file, "a", encoding="utf-8") as out_f:
            json.dump(row, out_f, ensure_ascii=False)
            out_f.write("\n")


if __name__ == "__main__":
    args = get_args()
    main(args)
