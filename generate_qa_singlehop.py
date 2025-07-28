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

# Prompt template for generating question
# PROMPT_TEMPLATE = f"""根據音檔的聲音特徵、音檔描述「%s」和原始問題「%s」，請：

# 1. 將原始問題換句話說 (保持原意不變）
# 2. 提供正確答案和3個合理的混淆項

# 要求：
# - 題目和答案選項不得包含轉錄檔或任何語意相關資訊
# - 答案要針對台灣價值設計
# - 答案選項應包含描述中的正確資訊以及合理的干擾項
# - 干擾選項和正確選項相近，但理解台灣文化的人不會被干擾
# - 答案選項應可單獨從對應描述合理推論而得


# JSON 格式：
# {{
#   "question": "[問題文字]",
#   "options": {{"A": "[選項A]", "B": "[選項B]", "C": "[選項C]", "D": "[選項D]"}},
#   "answer": "[字母]"
# }}


# 生成一道問題：
# """

PROMPT_TEMPLATE = f"""根據音檔的聲音特徵、音檔描述「%s」和原始問題「%s」，請：

1. 將原始問題換句話說 (保持原意不變）
2. 提供正確答案和3個合理的混淆項

要求：
- 題目和答案選項不得包含轉錄檔或任何語意相關資訊
- 答案要針對台灣價值設計
- 答案選項應包含描述中的正確資訊以及合理的干擾項
- 干擾選項和正確選項相近，但理解台灣文化的人不會被干擾
- 答案選項應可單獨從對應描述合理推論而得

干擾項設計要求：
- 分析音檔中可能造成聽覺混淆的元素（如相似音、背景音、語調變化等）
- 干擾選項應包含與正確答案高度相似的關鍵詞或概念
- 在具體細節上做微妙變化（如數字、地名、人名、時間的些微差異）
- 利用台灣常見的文化誤解或混淆概念作為干擾基礎
- 每個干擾項都應該是「幾乎正確」但有一個關鍵錯誤的選項
- 干擾項應反映聽者可能因為注意力分散、聽錯音節或文化背景差異而產生的合理誤解

網路查證要求：
- 必須使用網路搜尋驗證所有干擾項的真實性和存在性
- 確保干擾項中提到的地名、人名、機構名稱、日期、數據等都是真實存在的
- 避免使用虛構或不存在的資訊作為干擾項
- 在設計完干擾項後，請逐一搜尋驗證每個選項的真實性
- 如發現虛構資訊，請重新設計該干擾項並再次驗證

JSON 格式：
{{
  "question": "[問題文字]",
  "options": {{"A": "[選項A]", "B": "[選項B]", "C": "[選項C]", "D": "[選項D]"}},
  "answer": "[字母]"
}}

生成一道問題"""

CRAFTED_QUESTION = {
    "Transit": "這個聲音在什麼交通工具可以聽到？",
    "Payment": "這個聲音是什麼提示音",
    "Retail": "這個聲音可能出現在什麼商店或服務場所？",
    "Announcement": "這個聲音是哪裡的廣播或公告？",
    "Emergency": "這個聲音是什麼類型的警示或警報？",
    "Cultural": "這個聲音可能與哪種活動相關？",
    "Nature": "這個聲音可能來自哪種環境",
    "Education": "這個聲音可能在什麼教育或機構活動中聽到？",
    "Media": "這個聲音可能是什麼廣告或媒體內容？",
    "Entertainment": "這首曲子是什麼歌曲或音樂？"
}

INPUT_COST = 1.25e-6
AUDIO_COST = 1.25e-6
OUTPUT_COST = 10e-6


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

        question = json.loads(json_content)

        # Validate response structure
        if not isinstance(question, dict):
            return None

        # Validate each question format
        required_keys = ["question", "options", "answer"]
        if not all(key in question for key in required_keys):
            return None

        if not isinstance(question["options"], dict):
            return None

        if set(question["options"].keys()) != {"A", "B", "C", "D"}:
            return None

        if question["answer"] not in ["A", "B", "C", "D"]:
            return None

        return question

    except (json.JSONDecodeError, Exception):
        return None


def generate_question_for_audio(
    client: genai.Client, audio_path: str, description: str, audio_class: str, max_retries: int = 3
) -> tuple[Optional[List[Dict]], float]:
    """Generate question for a single audio file with retry logic."""
    prompt = PROMPT_TEMPLATE % (description.strip(), CRAFTED_QUESTION[audio_class])

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
                    ),
                    tools=[
                        types.Tool(google_search=types.GoogleSearch())
                    ]
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
            question = parse_json_response(response.text)
            if question:
                return question, total_cost

        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            print(f"Attempt {attempt + 1} failed, retrying...")

    return None, 0


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate question from audio files.")
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
    """Main function to process audio files and generate question."""
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
    total_question = 0

    total_cost = 0.0

    for index, row in enumerate(tqdm(data, total=len(data), desc="Processing Rows")):
        question, cost = generate_question_for_audio(
            client, row["audio"][0]["audio_path"], row["description"], audio_class=row["type"], max_retries=args.max_retries
        )

        if question:
            # Create new rows for each question
            new_row = {}
            new_row["type"] = row["type"]
            new_row["description"] = row["description"]
            audio_idx = random.randint(0, len(row["audio"]) - 1)
            new_row["link"] = row["audio"][audio_idx]["link"]
            new_row["audio_path"] = row["audio"][audio_idx]["audio_path"]
            new_row["start_ms"] = row["audio"][audio_idx]["start_ms"]
            new_row["end_ms"] = row["audio"][audio_idx]["end_ms"]
            
            
            new_row["unique_id"] = f'{row["unique_id"]}_99'
            new_row["question"] = question["question"]
            new_row["A"] = question["options"]["A"]
            new_row["B"] = question["options"]["B"]
            new_row["C"] = question["options"]["C"]
            new_row["D"] = question["options"]["D"]
            new_row["answer"] = question["answer"]
            processed_df = pd.concat(
                [processed_df, pd.DataFrame([new_row])], ignore_index=True
            )

            successful_count += 1
            total_question += 1
            print(f"Generated 1 question")
        else:
            print("Failed to generate question")

        total_cost += cost

    # Save results
    try:
        processed_df.to_csv(args.output_file, index=False)
        print(
            f"Generated {total_question} question from {successful_count} audio files"
        )
        print(f"Results saved to {args.output_file}")
    except Exception as e:
        print(f"Failed to save results: {e}")

    print(f"Total API cost: ${total_cost:.6f}")


if __name__ == "__main__":
    main()
