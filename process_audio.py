import csv
import argparse
import os
import sys
import uuid
import pandas as pd
from tqdm import tqdm

from utils import download_from_google_drive, download_from_yt, crop_audio

MAX_AUDIO_LENGTH = 15  # seconds
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
        help="Path to the output CSV file where processed audio data will be saved.",
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


def process_audio_download(row, args):
    """Helper function to download audio for a single row."""
    if "drive" in row["link"]:
        return download_from_google_drive(
            row["link"],
            format=args.format,
            output_path=args.audio_dir,
            output_id=row["unique_id"],
        )
    elif "youtu" in row["link"]:
        return download_from_yt(
            row["link"].split("&")[0],  # Remove any additional parameters
            format=args.format,
            output_path=args.audio_dir,
            output_id=row["unique_id"],
        )
    else:
        print(f"Unsupported link format: {row['link']}", file=sys.stderr)
        return None


def main(args):

    os.makedirs(args.audio_dir, exist_ok=True)

    # Read CSV file as DataFrame
    df = pd.read_csv(args.input_file, encoding="utf-8")

    # Generate unique IDs for each entry
    df["unique_id"] = [str(uuid.uuid4().hex)[:8].upper() for _ in range(len(df))]

    # Initialize audio_path column
    df["audio_path"] = None

    # Process each row with progress bar
    tqdm.pandas(desc="Processing audio files")
    df["audio_path"] = df.progress_apply(
        lambda row: process_audio_download(row, args), axis=1
    )

    if "start" in df.columns:
        df["start_ms"] = df["start"].apply(timestamp_to_ms)

    if "end" in df.columns:
        df["end_ms"] = df["end"].apply(timestamp_to_ms)

    # Crop audio files if necessary
    tqdm.pandas(desc="Cropping audio files")
    df[["audio_path", "start_ms", "end_ms"]] = df.progress_apply(
        lambda row: crop_audio(
            row["audio_path"],
            start_ms=row.get("start_ms", 0),
            end_ms=row.get("end_ms", MAX_AUDIO_LENGTH * 1000),
            output_format=args.format,
            max_length=MAX_AUDIO_LENGTH * 1000,
        ),
        axis=1,
    ).apply(pd.Series)
    # Drop start, end, link columns
    df.drop(columns=["start", "end"], inplace=True, errors="ignore")

    # print(f"DataFrame shape: {df.shape}")
    # print(f"Data type: {type(df)}")

    # Save processed data to output file
    df.to_csv(args.output_file, index=False, encoding="utf-8")
    print(f"Processed data saved to: {args.output_file}")


if __name__ == "__main__":
    args = get_args()
    main(args)
