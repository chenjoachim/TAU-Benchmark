import os
import json
import argparse

ANNOTATORS = ["伊甯", "俞蓁", "家愷", "思齊", "悅媗", "禹融", "紫渝", "致堯", "郁玲"]

def parse_args():
    parser = argparse.ArgumentParser(description="Add new columns to JSONL file")
    parser.add_argument("-i", "--input_file", type=str, help="Path to the input JSONL file")
    parser.add_argument("-s", "--suggestions_dir", type=str, help="Path to dir containing the suggestions JSON file")
    parser.add_argument("-o", "--output_file", type=str, help="Path to the output JSONL file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    suggestions = {}
    # Load suggestions from the provided JSON file
    for annotator in ANNOTATORS:
        with open(os.path.join(args.suggestions_dir, f"{annotator}.jsonl"), 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    suggestion = json.loads(line)
                except json.JSONDecodeError:
                    print(line)
                    raise Exception("JSON Decode Error")
                    continue
                unique_id = suggestion.get("uniqueId")
                if unique_id:
                    suggestions[unique_id] = {"question": suggestion.get("question", ""), "options": suggestion.get("options", []), "annotator_1": annotator, "suggestion_1": suggestion.get("suggestion", "")}

    with open(args.input_file, 'r', encoding='utf-8') as input_file, \
         open(args.output_file, 'w', encoding='utf-8') as output_file:

        for line in input_file:
            row = json.loads(line)
            unique_id = row.get("uniqueId")
            if unique_id in suggestions:
                row.update(suggestions[unique_id])
            else:
                row.update({"annotator_1": "", "suggestion_1": ""})
            output_file.write(json.dumps(row, ensure_ascii=False) + '\n')
