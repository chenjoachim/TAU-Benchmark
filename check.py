import os
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Check distribution of JSONL entries among annotators")
    parser.add_argument("-d", "--directory", type=str, help="Path to the directory containing annotator JSONL files")
    return parser.parse_args()

if __name__ == "__main__":
    # Check if each entry is assigned to exactly two annotators and count entries per annotator
    args = parse_args()
    annotator_files = [f for f in os.listdir(args.directory) if f.endswith('.jsonl')]
    annotator_entries = {}
    for file in annotator_files:
        with open(os.path.join(args.directory, file), 'r', encoding='utf-8') as f:
            entries = [json.loads(line) for line in f if line.strip()]
            annotator_entries[file] = entries

    entry_count = {}
    for annotator, entries in annotator_entries.items():
        entry_count[annotator] = len(entries)
        for entry in entries:
            entry_id = entry.get("uniqueId")
            if entry_id not in entry_count:
                entry_count[entry_id] = 0
            entry_count[entry_id] += 1
        # Check if there is no duplicate entries for the same annotator
        if len(entries) != len(set(e.get("uniqueId") for e in entries)):
            print(f"Duplicate entries found for annotator file: {annotator}")
        print(len(entries), len(set(e.get("uniqueId") for e in entries)))

    # Print total entries count of all annotators
    total_entries = sum(entry_count[annotator] for annotator in annotator_entries)
    print(f"Total entries assigned to annotators: {total_entries}")