import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Give report to accuracy and F1 score.")
    parser.add_argument("-i", "--input_file", help="Path to the input JSONL file.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    with open(args.input_file, "r", encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    print(f"Loaded {len(data)} entries from {args.input_file}")
    
    correct_count = sum(1 for entry in data if entry["correct"])
    accuracy = correct_count / len(data)
    print(f"Accuracy: {accuracy:.4f}")