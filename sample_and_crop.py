import json
import os
import argparse
from utils import (
    download_from_google_drive,
    download_from_yt,
    download_from_curl
)
from tqdm import tqdm
from pydub import AudioSegment

SAMPLING_RATE = 44100
MAX_AUDIO_LENGTH = 30 * 1000  # in milliseconds

def parse_args():
    parser = argparse.ArgumentParser(description="Process audio files and save to JSON.")
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSONL file containing audio file paths.",
    )
    parser.add_argument(
        "-d",
        "--audio_dir",
        type=str,
        default="data/raw",
        help="Directory where raw audio file are stored (default: data/raw).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/new",
        help="Directory where processed audio files will be saved (default: data/new).",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        default="data/processed.jsonl",
        help="Path to the output JSONL file where processed audio metadata will be saved.",
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
    os.makedirs(args.output_dir, exist_ok=True)
    
    for row in tqdm(data):
        save_path = os.path.join(args.audio_dir, row['audioPath'].split("/")[-1])

        # Resample to sr
        audio = AudioSegment.from_file(save_path)
        audio = audio.set_frame_rate(SAMPLING_RATE)
        
        # Crop according to start_ms and end_ms
        start_ms = row.get('startMs', 0)
        end_ms = row.get('endMs', len(audio))
        if start_ms < 0:
            start_ms = 0
            print(f"Warning: start_ms < 0 for {save_path}, setting to 0")
        if end_ms < 0 or end_ms - start_ms > MAX_AUDIO_LENGTH:
            end_ms = min(start_ms + MAX_AUDIO_LENGTH, len(audio))
            print(f"Warning: end_ms < 0 or too long for {save_path}, setting to {end_ms}")

        new_row = row.copy()
        new_row['startMs'] = start_ms
        new_row['endMs'] = end_ms
        output_data.append(new_row)
        audio = audio[start_ms:end_ms]
        output_path = os.path.join(args.output_dir, row['audioPath'].split("/")[-1])
        audio.export(output_path, format="mp3")
        
    with open(args.output_file, 'w') as f:
        for item in output_data:
            f.write(json.dumps(item) + "\n")