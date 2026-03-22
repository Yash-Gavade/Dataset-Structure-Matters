import json
import os
from typing import Dict, List

from datasets import Dataset, load_dataset

OUTPUT_DIR = "finetuning_data"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def format_example(instruction: str, input_text: str, output_text: str) -> str:
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    output_text = (output_text or "").strip()

    if input_text:
        prompt = (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{input_text}\n\n"
            "### Response:\n"
            f"{output_text}"
        )
    else:
        prompt = (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Response:\n"
            f"{output_text}"
        )
    return prompt


def prepare_alpaca(limit: int = 10000) -> Dataset:
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.select(range(min(limit, len(ds))))

    rows: List[Dict] = []
    for ex in ds:
        text = format_example(
            ex.get("instruction", ""),
            ex.get("input", ""),
            ex.get("output", "")
        )
        rows.append({
            "dataset": "alpaca",
            "instruction": ex.get("instruction", ""),
            "input": ex.get("input", ""),
            "output": ex.get("output", ""),
            "text": text,
        })
    return Dataset.from_list(rows)


def prepare_dolly(limit: int = 10000) -> Dataset:
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    ds = ds.select(range(min(limit, len(ds))))

    rows: List[Dict] = []
    for ex in ds:
        text = format_example(
            ex.get("instruction", ""),
            ex.get("context", ""),
            ex.get("response", "")
        )
        rows.append({
            "dataset": "dolly",
            "instruction": ex.get("instruction", ""),
            "input": ex.get("context", ""),
            "output": ex.get("response", ""),
            "text": text,
        })
    return Dataset.from_list(rows)


def prepare_oasst(limit: int = 10000) -> Dataset:
    ds = load_dataset("OpenAssistant/oasst1", split="train")

    rows: List[Dict] = []
    count = 0

    # simple single-turn extraction:
    # keep assistant messages with a parent prompt text when available
    by_id = {ex["message_id"]: ex for ex in ds}

    for ex in ds:
        if ex.get("role") != "assistant":
            continue

        parent_id = ex.get("parent_id")
        if not parent_id or parent_id not in by_id:
            continue

        parent = by_id[parent_id]
        if parent.get("role") != "prompter":
            continue

        instruction = parent.get("text", "").strip()
        output_text = ex.get("text", "").strip()

        if not instruction or not output_text:
            continue

        text = format_example(instruction, "", output_text)
        rows.append({
            "dataset": "oasst1",
            "instruction": instruction,
            "input": "",
            "output": output_text,
            "text": text,
        })
        count += 1
        if count >= limit:
            break

    return Dataset.from_list(rows)


def save_dataset(ds: Dataset, name: str) -> None:
    ensure_dir(OUTPUT_DIR)
    out_path = os.path.join(OUTPUT_DIR, name)
    ds.save_to_disk(out_path)

    preview_path = os.path.join(OUTPUT_DIR, f"{name}_preview.jsonl")
    with open(preview_path, "w", encoding="utf-8") as f:
        for ex in ds.select(range(min(5, len(ds)))):
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main() -> None:
    alpaca = prepare_alpaca(limit=10000)
    dolly = prepare_dolly(limit=10000)
    oasst = prepare_oasst(limit=10000)

    save_dataset(alpaca, "alpaca")
    save_dataset(dolly, "dolly")
    save_dataset(oasst, "oasst1")

    print("Saved datasets:")
    print(" - finetuning_data/alpaca")
    print(" - finetuning_data/dolly")
    print(" - finetuning_data/oasst1")


if __name__ == "__main__":
    main()