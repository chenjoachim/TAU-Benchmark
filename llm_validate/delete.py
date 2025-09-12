import os
import json
import argparse
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser(description="Delete too easy entries from a JSONL file.")
    parser.add_argument("-i", "--input_file", help="Path to the input JSONL file.")
    parser.add_argument("-l", "--llama_inference", help="Path to the Llama inference file.")
    parser.add_argument("-o", "--output_file", help="Path to the output JSONL file.")
    parser.add_argument("--deleted_file", help="Path to save deleted entries (optional).", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.llama_inference, "r", encoding='utf-8') as f:
        llama_inference = [json.loads(line) for line in f]

    with open(args.input_file, "r", encoding='utf-8') as f:
        input_data = [json.loads(line) for line in f]

    print(f"Loaded {len(input_data)} entries from {args.input_file}")
    
    correct_count = Counter()
    for entry in llama_inference:
        if entry["correct"]:
            correct_count[entry["uniqueId"]] += 1

    # Filter entries that are too easy (correct count >= 4)
    too_easy_ids = {id for id, count in correct_count.items() if count >= 4}

    # Write the filtered entries to the output file
    with open(args.output_file, "w", encoding='utf-8') as f:
        for entry in input_data:
            if entry["uniqueId"] not in too_easy_ids:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Filtered data written to {args.output_file}. Total entries: {len(input_data) - len(too_easy_ids)}")

    if args.deleted_file:
        with open(args.deleted_file, "w", encoding='utf-8') as f:
            for entry in input_data:
                if entry["uniqueId"] in too_easy_ids:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Deleted entries written to {args.deleted_file}. Total deleted: {len(too_easy_ids)}")