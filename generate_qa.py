import json
import os
from typing import List, Dict, Optional
import argparse

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Prompt template for generating questions
QUESTION_PER_AUDIO = 4
PROMPT_TEMPLATE = f"""為音檔生成 {QUESTION_PER_AUDIO} 道測試題，測試模型能否從音檔中推斷出「%s」這個描述。

目標：
- 測試模型能否從音檔的聲音特徵判斷出描述中的資訊
- 專注於文化特定的聲音、環境音、情境特徵
- 答案選項應包含描述中的正確資訊以及合理的干擾項

JSON 格式：
{{
  "question": "[問題文字]",
  "options": {{"A": "[選項A]", "B": "[選項B]", "C": "[選項C]", "D": "[選項D]"}},
  "answer": "[字母]"
}}

範例（假設音檔描述是「全聯福利中心的廣播聲」）：
{{
  "question": "根據這段音檔，此廣播最可能來自哪個場所？",
  "options": {{"A": "全聯福利中心", "B": "7-ELEVEN", "C": "傳統市場", "D": "百貨公司"}},
  "answer": "A"
}}

生成 {QUESTION_PER_AUDIO} 道問題：
"""


def parse_json_response(response_text: str) -> Optional[List[Dict]]:
    """Parse and validate JSON response from the API."""
    try:
        # Extract JSON from response
        if "```json" in response_text:
            json_content = (
                response_text.strip()
                .split("```json")[1]
                .strip()
                .split("```")[0]
                .strip()
            )
        else:
            json_content = response_text.strip()

        questions = json.loads(json_content)

        # Validate response structure
        if not isinstance(questions, list) or len(questions) != QUESTION_PER_AUDIO:
            return None

        # Validate each question format
        for question in questions:
            required_keys = ["question", "options", "answer"]
            if not all(key in question for key in required_keys):
                return None

            if not isinstance(question["options"], dict):
                return None

            if set(question["options"].keys()) != {"A", "B", "C", "D"}:
                return None

            if question["answer"] not in ["A", "B", "C", "D"]:
                return None

        return questions

    except (json.JSONDecodeError, Exception):
        return None


def generate_questions_for_audio(
    client: genai.Client, audio_path: str, description: str, max_retries: int = 3
) -> Optional[List[Dict]]:
    """Generate questions for a single audio file with retry logic."""
    prompt = PROMPT_TEMPLATE % description

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

            questions = parse_json_response(response.text)
            if questions:
                return questions

        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            print(f"Attempt {attempt + 1} failed, retrying...")

    return None


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
    output_columns = df.columns.tolist() + ["question", "A", "B", "C", "D", "answer"]
    processed_df = pd.DataFrame(columns=output_columns)

    successful_count = 0
    total_questions = 0

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Rows"):
        questions = generate_questions_for_audio(
            client, row["audio_path"], row["description"], max_retries=args.max_retries
        )

        if questions:
            # Create new rows for each question
            for idx, question_data in enumerate(questions):
                new_row = row.copy()
                new_row["unique_id"] = f'{row["unique_id"]}_{idx:02d}'
                new_row["question"] = question_data["question"]
                new_row["A"] = question_data["options"]["A"]
                new_row["B"] = question_data["options"]["B"]
                new_row["C"] = question_data["options"]["C"]
                new_row["D"] = question_data["options"]["D"]
                new_row["answer"] = question_data["answer"]
                processed_df = pd.concat(
                    [processed_df, pd.DataFrame([new_row])], ignore_index=True
                )

            successful_count += 1
            total_questions += len(questions)
            print(f"Generated {len(questions)} questions")
        else:
            print("Failed to generate questions")
            

    # Save results
    try:
        processed_df.to_csv(args.output_file, index=False)
        print(
            f"Generated {total_questions} questions from {successful_count} audio files"
        )
        print(f"Results saved to {args.output_file}")
    except Exception as e:
        print(f"Failed to save results: {e}")


if __name__ == "__main__":
    main()
