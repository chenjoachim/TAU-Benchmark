import json
import csv
import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Convert CSV to JSONL")
    parser.add_argument("-i", "--input_file", type=str, help="Path to the input CSV file")
    parser.add_argument("-o", "--output_file", type=str, help="Path to the output JSONL file")
    return parser.parse_args()

def main(args):
    with open(args.input_file, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        data = [row for row in csv_reader]

    with open(args.output_file, 'w', encoding='utf-8') as jsonl_file:
        for row in data:
            new_row = {}
            new_row["type"] = row["type"]
            new_row["description"] = row["description"]
            new_row["link"] = row["link"]
            new_row["uniqueId"] = row["unique_id"]
            new_row["audioPath"] = "https://huggingface.co/datasets/chenjoachim/TAU-dataset/resolve/main/" + row["audio_path"].split('/')[-1]
            new_row["startMs"] = int(row["start_ms"])
            new_row["endMs"] = int(row["end_ms"])
            new_row["question"] = row["question"]
            # Shuffle the options
            options = [row["A"], row["B"], row["C"], row["D"]]
            answer_text = row[row["answer"]]
            random.shuffle(options)
            new_row["options"] = options
            answer_index = options.index(answer_text)
            new_row["answer"] = chr(ord('A') + answer_index)
            jsonl_file.write(json.dumps(new_row, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    args = parse_args()
    main(args)