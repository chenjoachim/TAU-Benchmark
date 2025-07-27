import json
import os
import random
import time
from typing import List, Dict, Optional
import argparse

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from google import genai
from google.genai import types

PROMPT_TEMPLATE = """根據音檔，回答以下單選題：
問題：{question}
(A) {option_a}
(B) {option_b}
(C) {option_c}
(D) {option_d}
請選擇正確答案的字母（A、B、C 或 D）
"""


def evaluate(
    client: genai.Client, row: dict, max_retries: int = 3
) -> str:
    """Generate question for a single audio file with retry logic."""
    prompt = PROMPT_TEMPLATE.format(
        question=row["question"],
        option_a=row["A"],
        option_b=row["B"],
        option_c=row["C"],
        option_d=row["D"],
    )

    # Read audio file
    try:
        with open(row["audio_path"], "rb") as audio_file:
            audio_content = audio_file.read()
    except Exception:
        return None

    # Retry logic for API calls
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[
                    prompt,
                    types.Part.from_bytes(
                        data=audio_content,
                        mime_type="audio/mp3",
                    ),
                ],
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=8192,
                    )
                ),
            )
            question = response.text.strip()
            if question:
                return question

        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            print(f"Attempt {attempt + 1} failed, retrying...")
            time.sleep(2 ** attempt)  # Exponential backoff

    return ""


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate questions from audio files.")
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        required=True,
        help="Path to the input CSV file containing audio file paths.",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        required=True,
        help="Path to the output CSV file where processed audio data will be saved.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Number of retries for API calls in case of failure (default: 3).",
    )
    return parser.parse_args()


def main():
    """Main function to process audio files and generate questions."""
    args = parse_args()
    print("Starting question generation process...")

    load_dotenv()

    # Initialize GenAI client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found in environment variables")
        return

    client = genai.Client(api_key=api_key)

    # Load input data
    df = pd.read_csv(args.input_file)
    print(f"Loaded {len(df)} audio files from {args.input_file}")

    # Initialize output DataFrame
    output_columns = ['audio_path', 'description', 'question', 'A', 'B', 'C', 'D', 'answer', 'prediction']
    processed_df = pd.DataFrame(columns=output_columns)

    successful_count = 0
    total_questions = 0

    total_cost = 0.0

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Rows"):
        new_row = row.to_dict()
        # Random shuffle the options
        correct_text = new_row[new_row["answer"]]
        choices = [new_row[k] for k in "ABCD"]
        random.shuffle(choices)

        # Reassign and update answer
        for i, key in enumerate("ABCD"):
            new_row[key] = choices[i]
            if choices[i] == correct_text:
                new_row["answer"] = key

        prediction = evaluate(client, new_row, max_retries=args.max_retries)
        new_row['prediction'] = prediction
        # Keep only the relevant columns
        new_row = {col: new_row[col] for col in output_columns if col in new_row}
        processed_df = pd.concat([processed_df, pd.DataFrame([new_row])], ignore_index=True)

        # break

    # Save results
    try:
        processed_df.to_csv(args.output_file, index=False)
        print(
            f"Generated {total_questions} questions from {successful_count} audio files"
        )
        print(f"Results saved to {args.output_file}")
    except Exception as e:
        print(f"Failed to save results: {e}")

    print(f"Total API cost: ${total_cost:.6f}")


if __name__ == "__main__":
    main()
