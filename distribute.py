import json
import os
import argparse
import random

ANNOTATORS = ["俞蓁", "家愷", "思齊", "悅媗", "禹融", "紫渝", "致堯", "郁玲"]

def parse_args():
    parser = argparse.ArgumentParser(description="Distribute JSONL entries to annotators")
    parser.add_argument("-i", "--input_file", type=str, help="Path to the input JSONL file")
    parser.add_argument("-o", "--output_dir", type=str, help="Path to the output directory for annotator files")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Equally, randomly distribute entries to annotators
    entries = []
    with open(args.input_file, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            if line.strip():
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line}")
                    continue

    random.shuffle(entries)  # Shuffle entries to ensure random distribution
    annotator_idx = 0
    for entry in entries:
        entry_cnt = 2
        while entry_cnt > 0:
            if ANNOTATORS[annotator_idx % len(ANNOTATORS)] == entry.get("annotator_1", ""):
                annotator_idx += 1
                continue
            annotator = ANNOTATORS[annotator_idx % len(ANNOTATORS)]
            with open(os.path.join(args.output_dir, f"{annotator}.jsonl"), '+a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                
            annotator_idx += 1
            entry_cnt -= 1
