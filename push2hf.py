import pandas as pd
from datasets import Dataset, DatasetDict, Audio, Features, Value
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Push datasets to Hugging Face Hub",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="chenjoachim/TAU-Benchmark-tiny-flash",
        help="The repository ID to push the datasets to on Hugging Face Hub.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input CSV file containing the dataset.",
    )
    return parser.parse_args()

def main(args):
    # Load your CSV
    df = pd.read_csv(args.input_file)

    # Define the features schema
    features = Features({
        "unique_id": Value("string"),
        "audio": Audio(),
        "link": Value("string"),
        "start_ms": Value("int64"),
        "end_ms": Value("int64"),
        "question": Value("string"),
        "A": Value("string"),
        "B": Value("string"),
        "C": Value("string"),
        "D": Value("string"),
        "answer": Value("string"),
    })


    for type_name in df['type'].unique():
        type_df = df[df['type'] == type_name].copy()
        
        # Rename audio_path to audio and remove type column
        type_df = type_df.rename(columns={'audio_path': 'audio'})
        type_df = type_df.drop('type', axis=1)
        type_df = type_df.drop('description', axis=1)
        
        # Create dataset with explicit features
        dataset = Dataset.from_pandas(type_df, features=features, preserve_index=False)
        
        dataset.push_to_hub(args.repo_id, config_name=type_name, private=True)

if __name__ == "__main__":
    args = parse_args()
    main(args)