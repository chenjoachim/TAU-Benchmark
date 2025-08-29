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

AUDIO_DIR = "data/new"

PRICING = {
    "gemini-2.5-pro":{
        "input": 1.25e-6,
        "output": 1e-5,
        "audio_input": 1.25e-6,
    },
    "gemini-2.5-flash":{
        "input": 0.3e-6,
        "output": 2.5e-6,
        "audio_input": 1e-6,
    }
}


def evaluate(
    client: genai.Client, row: dict, model_name: str, max_retries: int = 3,
) -> tuple[str, float]:
    """Generate question for a single audio file with retry logic."""
    prompt = PROMPT_TEMPLATE.format(
        question=row["question"],
        option_a=row["options"][0],
        option_b=row["options"][1],
        option_c=row["options"][2],
        option_d=row["options"][3],
    )

    audio_path = os.path.join(AUDIO_DIR, row["audioPath"].split("/")[-1])
    # Read audio file
    try:
        with open(audio_path, "rb") as audio_file:
            audio_content = audio_file.read()
    except Exception:
        return None

    # Retry logic for API calls
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
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
                        # include_thoughts=True
                    )
                ),
            )
            answer = response.text.strip()
            output_cost = (
                response.usage_metadata.candidates_token_count
                + response.usage_metadata.thoughts_token_count
            ) * PRICING[model_name]["output"]
            input_cost = 0
            for modality in response.usage_metadata.prompt_tokens_details:
                if modality.modality == types.Modality.AUDIO:
                    input_cost += modality.token_count * PRICING[model_name]["audio_input"]
                elif modality.modality == types.Modality.TEXT:
                    input_cost += modality.token_count * PRICING[model_name]["input"]
            total_cost = input_cost + output_cost

            if answer:
                return answer, total_cost
            else:
                raise ValueError("Empty response")

        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            print(f"Attempt {attempt + 1} failed, retrying...")
            time.sleep(2 ** attempt)  # Exponential backoff

    return "", 0.0


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate questions from audio files.")
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSONL file containing audio file paths.",
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
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-2.5-pro",
        choices=["gemini-2.5-pro", "gemini-2.5-flash"],
        help="Model name to use for generation (default: gemini-2.5-pro).",
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

    data = []
    total_cost = 0.0

    # Load input data from JSONL file
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Avoid empty lines
                data.append(json.loads(line.strip()))
    
    for entry in tqdm(data, desc="Processing Entries"):
        answer, cost = evaluate(client, entry, model_name=args.model_name, max_retries=args.max_retries)
        entry["prediction"] = answer
        total_cost += cost
        with open(args.output_file, '+a', encoding='utf-8') as out_f:
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Total cost of API calls: ${total_cost:.6f}")


if __name__ == "__main__":
    main()
