import os

import pandas as pd

OUTPUT_PATH = "finetuned_models/analysis/manual_scoring_filled.csv"

rows = [
    [1, "explanation", "alpaca", 4, 4, 4, 4],
    [1, "explanation", "dolly", 3, 3, 3, 2],
    [1, "explanation", "oasst1", 3, 3, 4, 2],

    [2, "summarization", "alpaca", 2, 4, 4, 2],
    [2, "summarization", "dolly", 2, 3, 2, 2],
    [2, "summarization", "oasst1", 3, 4, 4, 3],

    [3, "coding", "alpaca", 2, 3, 2, 2],
    [3, "coding", "dolly", 2, 3, 3, 2],
    [3, "coding", "oasst1", 1, 1, 2, 1],

    [4, "reasoning", "alpaca", 1, 1, 3, 1],
    [4, "reasoning", "dolly", 1, 1, 1, 1],
    [4, "reasoning", "oasst1", 2, 1, 2, 1],

    [5, "translation", "alpaca", 2, 2, 2, 1],
    [5, "translation", "dolly", 2, 1, 2, 1],
    [5, "translation", "oasst1", 1, 1, 2, 1],

    [6, "creative", "alpaca", 3, 3, 3, 2],
    [6, "creative", "dolly", 2, 2, 2, 1],
    [6, "creative", "oasst1", 4, 4, 4, 4],
]

def main():
    df = pd.DataFrame(
        rows,
        columns=[
            "item_id",
            "category",
            "model",
            "instruction_following",
            "correctness",
            "clarity",
            "completeness",
        ],
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}")
    print(df)

if __name__ == "__main__":
    main()