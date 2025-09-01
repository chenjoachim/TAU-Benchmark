import json
import argparse
import copy

from vllm import LLM, SamplingParams

def parse_args():
    parser = argparse.ArgumentParser(description="Generate text using a pre-trained LLM model.")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Pre-trained model ID.")
    parser.add_argument("-i", "--input_file", type=str, required=False, help="Path to the input file containing prompts.")
    parser.add_argument("-o", "--output_file", type=str, required=False, help="Path to the output file to save generated texts.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.75, help="GPU memory utilization.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Initialize model
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=2048,
        max_num_seqs=args.batch_size,
    )

    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=192
    )
    
    data = []

    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))

    PROMPT_TEMPLATE = """音檔的逐字稿：「{transcript}」
根據此逐字稿，回答以下單選題：
問題：{question}
(A) {option_a}
(B) {option_b}
(C) {option_c}
(D) {option_d}
請選擇正確答案的字母（A、B、C 或 D）
"""

    # Batch prompts
    for item in data:
        item["prompt"] = PROMPT_TEMPLATE.format(
            transcript=item["transcription"],
            question=item["question"],
            option_a=item["options"][0],
            option_b=item["options"][1],
            option_c=item["options"][2],
            option_d=item["options"][3]
        )
    
    # Copy data 5 times
    new_data = []

    for item in data:
        for _ in range(5):
            new_data.append(copy.deepcopy(item))

    data = new_data

    prompts = [
        item["prompt"] for item in data
    ]

    # Generate responses
    outputs = llm.generate(prompts, sampling_params)

    for item, output in zip(data, outputs):
        item["generation"] = output.outputs[0].text.strip()

    with open(args.output_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
