import json
import os

import pandas as pd

FILES = {
    "alpaca": "finetuned_models/eval_alpaca.json",
    "dolly": "finetuned_models/eval_dolly.json",
    "oasst1": "finetuned_models/eval_oasst1.json",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    all_data = {}

    for model_name, path in FILES.items():
        all_data[model_name] = load_json(path)

    num_items = len(next(iter(all_data.values())))

    rows = []
    for i in range(num_items):
        prompt = all_data["alpaca"][i]["prompt"]
        category = all_data["alpaca"][i]["category"]

        row = {
            "item_id": i + 1,
            "category": category,
            "prompt": prompt,

            "alpaca_response": all_data["alpaca"][i]["response"],
            "alpaca_instruction_following": "",
            "alpaca_correctness": "",
            "alpaca_clarity": "",
            "alpaca_completeness": "",

            "dolly_response": all_data["dolly"][i]["response"],
            "dolly_instruction_following": "",
            "dolly_correctness": "",
            "dolly_clarity": "",
            "dolly_completeness": "",

            "oasst1_response": all_data["oasst1"][i]["response"],
            "oasst1_instruction_following": "",
            "oasst1_correctness": "",
            "oasst1_clarity": "",
            "oasst1_completeness": "",
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs("finetuned_models/analysis", exist_ok=True)
    out_path = "finetuned_models/analysis/manual_scoring_template.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()