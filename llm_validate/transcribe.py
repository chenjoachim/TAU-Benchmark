import json
import argparse

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Transcribe audio using a pre-trained model.")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to the input JSONL file containing audio paths.")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Path to the output JSONL file to save transcriptions.")
    parser.add_argument("--model_id", type=str, default="openai/whisper-large-v3", help="Pre-trained model ID.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing.")
    parser.add_argument("--data_dir", type=str, default="data/new", help="Directory containing audio files.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = args.model_id

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        lines = f.readlines()
    dataset = [json.loads(line.strip()) for line in lines]

    for i in tqdm(range(0, len(dataset), args.batch_size)):
        batch = dataset[i:i + args.batch_size]
        audio_paths = [f"{args.data_dir}/{item['audioPath'].split('/')[-1]}" for item in batch]
        results = pipe(audio_paths, batch_size=args.batch_size, generate_kwargs={"language": "zh", "max_new_tokens": 128})
        for item, result in zip(batch, results):
            item["transcription"] = result["text"]

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
