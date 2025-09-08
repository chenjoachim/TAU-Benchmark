import os
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Add hop tag to JSONL files")
    parser.add_argument("-i", "--input_dir", type=str, help="Path to the input directory containing JSONL files")
    parser.add_argument("-t", "--tag_dir", type=str, help="Directory containing hop tag JSON files")
    parser.add_argument("-o", "--output_dir", type=str, help="Path to the output directory for JSONL files")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    input_dir = args.input_dir
    tag_dir = args.tag_dir
    output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Build a mapping from uniqueId to hop tag
    hop_tags = {}
    for tag_file in os.listdir(tag_dir):
        if tag_file.endswith('.jsonl'):
            with open(os.path.join(tag_dir, tag_file), 'r', encoding='utf-8') as f:
                tag_data = [json.loads(line) for line in f if line.strip()]
            for item in tag_data:
                unique_id = item.get("uniqueId")
                hop_tag = item.get("hopType", "")
                if unique_id:
                    hop_tags[unique_id] = hop_tag
    
    print(hop_tags.items())     
    # Process each JSONL file in the input directory
    for input_file in os.listdir(input_dir):
        print('Processing file:', input_file)
        if input_file.endswith('.jsonl'):
            with open(os.path.join(input_dir, input_file), 'r', encoding='utf-8') as infile, \
                 open(os.path.join(output_dir, input_file), 'w', encoding='utf-8') as outfile:
                
                for line in infile:
                    row = json.loads(line)
                    unique_id = row.get("uniqueId")
                    hop_tag = hop_tags.get(unique_id, "")
                    if not hop_tag:
                        print(f"Warning: No hop tag found for uniqueId {unique_id}")
                    row["hopType"] = hop_tag
                    outfile.write(json.dumps(row, ensure_ascii=False) + '\n')
                    
    
