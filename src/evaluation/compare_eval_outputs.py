import json
import math
import os
from collections import Counter, defaultdict

import pandas as pd

FILES = {
    "alpaca": "finetuned_models/eval_alpaca.json",
    "dolly": "finetuned_models/eval_dolly.json",
    "oasst1": "finetuned_models/eval_oasst1.json",
}


def tokenize(text: str):
    return text.strip().split()


def shannon_entropy(tokens):
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log2(p)
    return entropy


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    rows = []
    category_rows = []

    for model_name, path in FILES.items():
        if not os.path.exists(path):
            print(f"Missing file: {path}")
            continue

        data = load_json(path)

        all_tokens = []
        cat_lengths = defaultdict(list)

        for item in data:
            response = item.get("response", "").strip()
            category = item.get("category", "unknown")

            tokens = tokenize(response)
            all_tokens.extend(tokens)
            cat_lengths[category].append(len(tokens))

        vocab = set(all_tokens)
        total_tokens = len(all_tokens)

        row = {
            "model": model_name,
            "num_outputs": len(data),
            "total_tokens": total_tokens,
            "avg_response_length": total_tokens / len(data) if data else 0.0,
            "vocab_size": len(vocab),
            "ttr": len(vocab) / total_tokens if total_tokens > 0 else 0.0,
            "entropy": shannon_entropy(all_tokens),
        }
        rows.append(row)

        for category, lengths in cat_lengths.items():
            category_rows.append({
                "model": model_name,
                "category": category,
                "avg_response_length": sum(lengths) / len(lengths) if lengths else 0.0,
                "num_examples": len(lengths),
            })

    overall_df = pd.DataFrame(rows)
    category_df = pd.DataFrame(category_rows)

    os.makedirs("finetuned_models/analysis", exist_ok=True)

    overall_path = "finetuned_models/analysis/eval_summary.csv"
    category_path = "finetuned_models/analysis/eval_category_lengths.csv"

    overall_df.to_csv(overall_path, index=False)
    category_df.to_csv(category_path, index=False)

    print("Saved:")
    print(overall_path)
    print(category_path)
    print()
    print(overall_df)


if __name__ == "__main__":
    main()