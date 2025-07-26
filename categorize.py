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
GUIDE = """大眾運輸信號聲 (Transit)
 定義：用於捷運、高鐵、台鐵、公車等交通工具的進／出站提示、門開關、警示及行駛聲響。
 範例：板南線進站音、中和新蘆線列車進站警示音、公車到站氣壓音、台北捷運即將關門高頻嗶聲、高鐵到站提示音。


票務／支付提示音 (Payment)
 定義：所有與刷卡、電子支付、票卡餘額、租借系統等交易流程相關的機器提示音。
 範例：捷運刷卡機「請退出門擋區，再感應票卡」、Youbike 2.0 租借聲、公車電子支付餘額不足、交易取消請取回現金及卡片。


商店與服務環境音樂／提示 (Retail)
 定義：便利商店、超市、零售通路及服務櫃檯常見的背景音樂、門響、結帳操作提示。
 範例：7‑11 開門音、全家結帳螢幕按鈕、萊爾富開門聲、全聯福利中心音樂、請支援收銀。


公共廣播與資訊導引 (Announcement)
 定義：各類公共空間（車站、機場、廟會、社區）中的人工或自動化廣播，用以報站、導引、公告活動。
 範例：機場登機廣播開頭、台中火車站月台廣播、廟會舞獅擲筊喊話、中元節普渡嗩吶聲。


緊急與安全警示 (Emergency)
 定義：用於突發事件或演習的警報音、警示語音，以及緊急車輛（救護、消防、警車）鳴笛。
 範例：空襲警報、地震簡訊警報、消防車鳴笛、宿舍火災警報。


文化／宗教與儀式音樂 (Cultural)
 定義：廟會、祭典、宗教儀式、傳統慶典中常聽到的鑼鼓、唸誦、法會唱佛等。
 範例：媽祖遶境敲鑼、唱佛聲、木魚聲、大悲咒、電音三太子、布袋戲叫喊。


自然生態與動物叫聲 (Nature)
 定義：台灣常見的鳥類、蛙鳴、壁虎、夜市環境中的自然音效。
 範例：台灣藍鵲叫聲、青蛙叫、壁虎聲、夜市人聲與機台混合聲、蚵仔煎鐵板聲。


教育／機構活動音 (Education)
 定義：校園、軍訓、補習班等機構裡的鐘聲、口令、聽力考試提示等教學活動音。
 範例：國小上下課鐘聲、軍訓「立正／稍息」口令、英聽「第1題至第4題，請聽…」、護眼操口令。


媒體廣告與名人配音 (Media)
 定義：廣告片段背景音效、名人台詞或配音，以及商品宣傳的標語音樂。
 範例：Uber Eats 叮咚音效、統一蜜豆奶廣告、蔡英文說話聲音、小S配音、Foodpanda 「叫foodpanda送」。


娛樂與休閒氛圍聲 (Entertainment)
 定義：遊樂場、遊戲機台、拍賣、演唱會、卡拉OK 等休閒娛樂相關的音效與音樂。
 範例：夾娃娃下爪聲、夜市射氣球聲、麻將聲、卡拉OK歡唱台語老歌、湯姆熊遊樂場環境音。
"""
PROMPT_TEMPLATE = f"""請根據以下分類指引對一個音檔的描述「%s」進行分類：

{GUIDE}

請分析音檔內容並按照以下JSON格式輸出：

```json
{{
  "category": "[分類名稱對應的英文單字]",
  "confidence": [1-10的整數]
}}
```
"""

CLASSES = ["Transit", "Payment", "Retail", "Announcement", "Emergency", "Cultural", "Nature", "Education", "Media", "Entertainment"]
INPUT_COST = 0.3e-6
AUDIO_COST = 1.0e-6
OUTPUT_COST = 2.5e-6


def parse_json_response(response_text: str) -> tuple[str, int]:
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

        answer = json.loads(json_content)

        if isinstance(answer, dict) and "category" in answer and "confidence" in answer:
            category = answer["category"]
            confidence = int(answer["confidence"])

            if category in CLASSES:
                return category, confidence
            else:
                print(f"Invalid category '{category}' in response.")
                return "", 0
        else:
            print("Invalid JSON structure or missing fields.")
            return "", 0

    except (json.JSONDecodeError, Exception):
        return "", 0


def generate_questions_for_audio(
    client: genai.Client, audio_path: str, description: str, max_retries: int = 3
) -> tuple[Optional[str], float]:
    """Generate questions for a single audio file with retry logic."""
    prompt = PROMPT_TEMPLATE % (description.strip())

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
            category, confidence = parse_json_response(response.text)

            if not category or confidence < 1 or confidence > 10:
                raise ValueError(
                    f"Invalid category '{category}' or confidence {confidence} in response."
                )

            if confidence <= 3:
                print(f"Low confidence ({confidence}) for audio {audio_path}, needing manual review.")
            
            return category, total_cost

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
        help="Path to the input JSONL file containing audio file paths.",
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

    total_cost = 0.0

    for index, row in enumerate(tqdm(data, total=len(data), desc="Processing Rows")):
        category, cost = generate_questions_for_audio(
            client, row["audio"][0]["audio_path"], row["description"], max_retries=args.max_retries
        )

        if category:
            with open(args.output_file, "a", encoding='utf-8') as f:
                row.update({
                    "type": category,
                })
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        total_cost += cost


    print(f"Total API cost: ${total_cost:.6f}")


if __name__ == "__main__":
    main()
