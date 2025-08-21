import json
import os
import argparse
from utils import (
    download_from_google_drive,
    download_from_yt,
    download_from_curl
)
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Process audio files and save to JSON.")
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSONL file containing audio file paths.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.input_file):
        print(f"Input file {args.input_file} does not exist.")
        exit(1)

    with open(args.input_file, 'r') as f:
        data = [json.loads(line) for line in f]

    output_data = []
    output_path = "raw"
    success_count = 0
    for row in tqdm(data):
        save_path = row['audioPath'].split("/")[-1]
        save_name = save_path.split(".")[0]
        output_id = save_name.split("_")[0]
        audio_idx = save_name.split("_")[1]
        
        if "drive" in row['link']:
            audio_path, success = download_from_google_drive(
                row['link'], output_path=output_path, output_id=output_id, audio_idx=audio_idx
            )
        elif "youtu" in row['link']:
            audio_path, success = download_from_yt(
                row['link'].split("&")[0], output_path=output_path, output_id=output_id, audio_idx=audio_idx
            )
        else:
            audio_path, success = download_from_curl(
                row['link'], output_path=output_path, output_id=output_id, audio_idx=audio_idx
            )
        
        if not audio_path:
            print(f"Failed to download audio for {row['link']}")
            continue
        
        success_count += 1
        
    print(f"Successfully downloaded {success_count} audio files.")
    print(f"Total files processed: {len(data)}")