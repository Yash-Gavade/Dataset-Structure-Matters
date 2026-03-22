import argparse
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--prompts_file", type=str, default="finetuning/eval_prompts.json")
    parser.add_argument("--output_file", type=str, required=True)
    return parser.parse_args()


def generate_response(model, tokenizer, prompt: str) -> str:
    full_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.replace(full_prompt, "").strip()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    with open(args.prompts_file, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    rows = []
    for item in prompts:
        response = generate_response(model, tokenizer, item["prompt"])
        rows.append({
            "id": item["id"],
            "category": item["category"],
            "prompt": item["prompt"],
            "response": response,
            "model_path": args.adapter_path
        })

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"Saved results to: {args.output_file}")


if __name__ == "__main__":
    main()