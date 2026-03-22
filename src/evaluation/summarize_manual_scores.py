import os

import pandas as pd

INPUT_PATH = "finetuned_models/analysis/manual_scoring_filled.csv"
OUTPUT_PATH = "finetuned_models/analysis/manual_score_summary.csv"

def main():
    df = pd.read_csv(INPUT_PATH)

    summary = (
        df.groupby("model")[["instruction_following", "correctness", "clarity", "completeness"]]
        .mean()
        .reset_index()
    )

    summary["overall_mean"] = summary[
        ["instruction_following", "correctness", "clarity", "completeness"]
    ].mean(axis=1)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(summary)

if __name__ == "__main__":
    main()