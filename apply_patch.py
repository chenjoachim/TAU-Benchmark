import json
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Apply patch to JSONL file.")
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        default="data/patched.jsonl",
        help="Path to the output JSONL file where patched data will be saved (default: data/patched.jsonl).",
    )
    parser.add_argument(
        "-p",
        "--patch_file",
        type=str,
        required=True,
        help="Path to the patch JSON file.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    patches = {}
    with open(args.patch_file, 'r', encoding='utf-8') as f:
        for line in f:
            patch = json.loads(line)
            patches[patch['audioPath']] = patch

    # Apply patches
    for item in data:
        audio_path = item.get('audioPath')
        if audio_path in patches:
            item['startMs'] = patches[audio_path]['startMs']
            item['endMs'] = patches[audio_path]['endMs']

    with open(args.output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Patched data saved to {args.output_file}")