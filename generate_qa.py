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
PROMPT_TEMPLATE = """Generate 3 multiple-choice questions in JSON format for audio: "%s".

Requirements:
- Test local/cultural sound recognition abilities
- Answers must be clearly derivable from the audio content
- Focus on culturally-specific sounds, environments, or contexts

JSON format:
{
  "question": "[Question text]",
  "options": {"A": "[Option A]", "B": "[Option B]", "C": "[Option C]", "D": "[Option D]"},
  "answer": "[Letter]"
}

Example:
{
  "question": "根據這段聲音，請問此廣播最有可能來自下列哪一個場所？",
  "options": {"A": "全聯福利中心", "B": "7-ELEVEN 便利商店", "C": "傳統菜市場", "D": "電影院售票口"},
  "answer": "A"
}

Generate 3 questions:
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
        if not isinstance(questions, list) or len(questions) != 3:
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

            questions = parse_json_response(response.text)
            if questions:
                return questions

        except Exception:
            pass

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
            for question_data in questions:
                new_row = row.copy()
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

        break  # For now, process only the first file (remove this break to process all files)

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
