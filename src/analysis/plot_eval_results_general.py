import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--category_csv", required=True)
    parser.add_argument("--out_dir", default="finetuned_models/analysis_v2")
    parser.add_argument("--tag", default="v2")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    summary_df = pd.read_csv(args.summary_csv)
    category_df = pd.read_csv(args.category_csv)

    # Avg response length
    plt.figure()
    plt.bar(summary_df["model"], summary_df["avg_response_length"])
    plt.title("Average Response Length by Model")
    plt.xlabel("Model")
    plt.ylabel("Average Response Length (tokens)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"avg_response_length_{args.tag}.png"), dpi=200)
    plt.close()

    # Vocab size
    plt.figure()
    plt.bar(summary_df["model"], summary_df["vocab_size"])
    plt.title("Vocabulary Size by Model")
    plt.xlabel("Model")
    plt.ylabel("Vocabulary Size")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"vocab_size_{args.tag}.png"), dpi=200)
    plt.close()

    # Entropy
    plt.figure()
    plt.bar(summary_df["model"], summary_df["entropy"])
    plt.title("Output Entropy by Model")
    plt.xlabel("Model")
    plt.ylabel("Shannon Entropy")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"output_entropy_{args.tag}.png"), dpi=200)
    plt.close()

    # Repeated line rate
    plt.figure()
    plt.bar(summary_df["model"], summary_df["avg_repeated_line_rate"])
    plt.title("Average Repeated Line Rate by Model")
    plt.xlabel("Model")
    plt.ylabel("Repeated Line Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"repeated_line_rate_{args.tag}.png"), dpi=200)
    plt.close()

    # Repeated trigram ratio
    plt.figure()
    plt.bar(summary_df["model"], summary_df["avg_repeated_trigram_ratio"])
    plt.title("Average Repeated Trigram Ratio by Model")
    plt.xlabel("Model")
    plt.ylabel("Repeated Trigram Ratio")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"repeated_trigram_ratio_{args.tag}.png"), dpi=200)
    plt.close()

    # Category-wise response length
    pivot_len = category_df.pivot(index="category", columns="model", values="avg_response_length")
    pivot_len.plot(kind="bar")
    plt.title("Average Response Length by Category")
    plt.xlabel("Category")
    plt.ylabel("Average Response Length (tokens)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"category_response_length_{args.tag}.png"), dpi=200)
    plt.close()

    # Category-wise entropy
    pivot_ent = category_df.pivot(index="category", columns="model", values="avg_response_entropy")
    pivot_ent.plot(kind="bar")
    plt.title("Average Response Entropy by Category")
    plt.xlabel("Category")
    plt.ylabel("Average Entropy")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"category_entropy_{args.tag}.png"), dpi=200)
    plt.close()

    print("Saved plots to:", args.out_dir)


if __name__ == "__main__":
    main()