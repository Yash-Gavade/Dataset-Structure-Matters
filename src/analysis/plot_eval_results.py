import os

import matplotlib.pyplot as plt
import pandas as pd

ANALYSIS_DIR = "finetuned_models/analysis"


def main():
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    summary_path = os.path.join(ANALYSIS_DIR, "eval_summary.csv")
    category_path = os.path.join(ANALYSIS_DIR, "eval_category_lengths.csv")

    summary_df = pd.read_csv(summary_path)
    category_df = pd.read_csv(category_path)

    # Overall average response length
    plt.figure()
    plt.bar(summary_df["model"], summary_df["avg_response_length"])
    plt.title("Average Response Length by Model")
    plt.xlabel("Model")
    plt.ylabel("Average Response Length (tokens)")
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, "avg_response_length.png"), dpi=200)
    plt.close()

    # Vocabulary size
    plt.figure()
    plt.bar(summary_df["model"], summary_df["vocab_size"])
    plt.title("Vocabulary Size by Model")
    plt.xlabel("Model")
    plt.ylabel("Vocabulary Size")
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, "vocab_size.png"), dpi=200)
    plt.close()

    # Entropy
    plt.figure()
    plt.bar(summary_df["model"], summary_df["entropy"])
    plt.title("Output Entropy by Model")
    plt.xlabel("Model")
    plt.ylabel("Shannon Entropy")
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, "output_entropy.png"), dpi=200)
    plt.close()

    # Per-category response length
    pivot_df = category_df.pivot(index="category", columns="model", values="avg_response_length")
    pivot_df.plot(kind="bar")
    plt.title("Average Response Length by Category")
    plt.xlabel("Category")
    plt.ylabel("Average Response Length (tokens)")
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, "category_response_length.png"), dpi=200)
    plt.close()

    print("Saved plots to:", ANALYSIS_DIR)


if __name__ == "__main__":
    main()