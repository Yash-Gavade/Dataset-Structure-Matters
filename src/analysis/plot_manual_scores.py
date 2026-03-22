import os
import pandas as pd
import matplotlib.pyplot as plt

INPUT_PATH = "finetuned_models/analysis/manual_score_summary.csv"
OUT_DIR = "finetuned_models/analysis"

def main():
    df = pd.read_csv(INPUT_PATH)

    metrics = ["instruction_following", "correctness", "clarity", "completeness"]

    # Grouped bar chart
    plot_df = df.set_index("model")[metrics]
    plot_df.plot(kind="bar")
    plt.title("Manual Evaluation Scores by Model")
    plt.xlabel("Model")
    plt.ylabel("Average Score")
    plt.ylim(0, 5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "manual_scores_by_model.png"), dpi=200)
    plt.close()

    # Overall mean chart
    plt.figure()
    plt.bar(df["model"], df["overall_mean"])
    plt.title("Overall Mean Manual Score")
    plt.xlabel("Model")
    plt.ylabel("Average Score")
    plt.ylim(0, 5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "overall_manual_score.png"), dpi=200)
    plt.close()

    print("Saved plots to:", OUT_DIR)

if __name__ == "__main__":
    main()