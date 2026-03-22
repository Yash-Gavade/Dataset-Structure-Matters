import os

import matplotlib.pyplot as plt
import pandas as pd

INPUT_PATH = "finetuned_models/analysis_constraints/constraint_summary.csv"
OUT_DIR = "finetuned_models/analysis_constraints"

def main():
    df = pd.read_csv(INPUT_PATH)

    plt.figure()
    plt.bar(df["model"], df["constraint_success_rate"])
    plt.title("Constraint-Following Success Rate")
    plt.xlabel("Model")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "constraint_success_rate.png"), dpi=200)
    plt.close()

    print("Saved plot to:", OUT_DIR)

if __name__ == "__main__":
    main()