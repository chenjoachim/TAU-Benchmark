import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Give report to accuracy and F1 score.")
    parser.add_argument("-i", "--input_file", help="Path to the input JSONL file.")
    parser.add_argument("--subset", type=str, default="", help="JSONL of the subset of the dataset to evaluate (e.g., 'train', 'dev', 'test').")
    parser.add_argument("-o", "--output_file", type=str, default="", help="Path to the output JSONL file if subset is specified.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    with open(args.input_file, "r", encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    if args.subset:
        with open(args.subset, "r", encoding='utf-8') as f:
            subset_ids = {json.loads(line)["uniqueId"] for line in f}
            data = [entry for entry in data if entry["uniqueId"] in subset_ids]

    print(f"Loaded {len(data)} entries from {args.input_file}")

    single_hop_count = sum(1 for entry in data if entry["hopType"] == "Single-hop")
    multi_hop_count = sum(1 for entry in data if entry["hopType"] == "Multi-hop")
    print(f"Single-hop questions: {single_hop_count}, Multi-hop questions: {multi_hop_count}")
    
    correct_count = sum(1 for entry in data if entry["correct"])
    single_hop_correct = sum(1 for entry in data if entry["correct"] and entry["hopType"] == "Single-hop")
    multi_hop_correct = sum(1 for entry in data if entry["correct"] and entry["hopType"] == "Multi-hop")
    accuracy = correct_count / len(data)
    single_hop_accuracy = single_hop_correct / single_hop_count if single_hop_count > 0 else 0
    multi_hop_accuracy = multi_hop_correct / multi_hop_count if multi_hop_count > 0 else 0
    print(f"Accuracy: {accuracy:.4f}, Single-hop Accuracy: {single_hop_accuracy:.4f}, Multi-hop Accuracy: {multi_hop_accuracy:.4f}")

    if args.output_file:
        with open(args.output_file, "w", encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
