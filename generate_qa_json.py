import json
import os
from typing import List, Dict, Optional
import argparse
import random

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Prompt template for generating questions
QUESTION_PER_AUDIO = 3
PROMPT_TEMPLATE = f"""請為音檔和對應的描述「%s」生成 {QUESTION_PER_AUDIO} 道測試題，測試使用者能否從音檔中推斷出對應的描述。

目標：
- 測試使用者聽到該台灣特色音檔是否能做出包含場景、用途、類型等聯想
- 題目和答案選項不得包含轉錄檔或任何語意相關資訊
- 答案選項應包含描述中的正確資訊以及合理的干擾項
- 干擾選項正確選項相近，但理解台灣文化的人不會被干擾
- 答案選項應可單獨從對應描述合理推論而得
- 每道題出題方向盡量相異
- 每道題和選項都必須加入台灣的情境和文化元素


JSON 格式：
{{
  "question": "[問題文字]",
  "options": {{"A": "[選項A]", "B": "[選項B]", "C": "[選項C]", "D": "[選項D]"}},
  "answer": "[字母]"
}}

範例（假設音檔描述是「全聯福利中心的廣播聲」）：
{{
  "question": "你最有可能在哪個場所聽到這個廣播？",
  "options": {{"A": "全聯福利中心", "B": "7-ELEVEN", "C": "傳統市場", "D": "百貨公司"}},
  "answer": "A"
}}

生成 {QUESTION_PER_AUDIO} 道問題：
"""

INPUT_COST = 0.3e-6
AUDIO_COST = 1.0e-6
OUTPUT_COST = 2.5e-6


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
) -> tuple[Optional[List[Dict]], float]:
    """Generate questions for a single audio file with retry logic."""
    prompt = PROMPT_TEMPLATE % (description.strip())

    print(prompt[:30])

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
                model="gemini-2.5-flash",
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
            output_cost = (
                response.usage_metadata.candidates_token_count
                + response.usage_metadata.thoughts_token_count
            ) * OUTPUT_COST
            input_cost = 0
            for modality in response.usage_metadata.prompt_tokens_details:
                if modality.modality == types.Modality.AUDIO:
                    input_cost += modality.token_count * AUDIO_COST
                elif modality.modality == types.Modality.TEXT:
                    input_cost += modality.token_count * INPUT_COST
            total_cost = input_cost + output_cost
            questions = parse_json_response(response.text)
            if questions:
                return questions, total_cost

        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            print(f"Attempt {attempt + 1} failed, retrying...")

    return None, 0


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
    with open(args.input_file, "r", encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    print(f"Loaded {len(data)} audio files from {args.input_file}")

    # Initialize output DataFrame
    output_columns = ["type", "description", "link", "unique_id", "audio_path", "start_ms", "end_ms", "question", "A", "B", "C", "D", "answer"]
    processed_df = pd.DataFrame(columns=output_columns)

    successful_count = 0
    total_questions = 0

    total_cost = 0.0

    for index, row in enumerate(tqdm(data, total=len(data), desc="Processing Rows")):
        questions, cost = generate_questions_for_audio(
            client, row["audio"][0]["audio_path"], row["description"], max_retries=args.max_retries
        )

        if questions:
            # Create new rows for each question
            for idx, question_data in enumerate(questions):
                new_row = {}
                new_row["type"] = row["type"]
                new_row["description"] = row["description"]
                audio_idx = idx % len(row["audio"])
                new_row["link"] = row["audio"][audio_idx]["link"]
                new_row["audio_path"] = row["audio"][audio_idx]["audio_path"]
                new_row["start_ms"] = row["audio"][audio_idx]["start_ms"]
                new_row["end_ms"] = row["audio"][audio_idx]["end_ms"]
                
                
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

        total_cost += cost

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
