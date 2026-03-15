from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd

from .text_utils import whitespace_tokenize


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def plot_length_histograms(df: pd.DataFrame, out_dir: str) -> None:
    _ensure_dir(out_dir)
    for ds, g in df.groupby("dataset"):
        instr_l = [len(whitespace_tokenize(x)) for x in g["instruction"].astype(str).tolist()]
        out_l = [len(whitespace_tokenize(x)) for x in g["output"].astype(str).tolist()]

        plt.figure()
        plt.hist(instr_l, bins=50)
        plt.title(f"Instruction length histogram — {ds}")
        plt.xlabel("Tokens (whitespace)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"hist_instr_len_{ds}.png"), dpi=200)
        plt.close()

        plt.figure()
        plt.hist(out_l, bins=50)
        plt.title(f"Output length histogram — {ds}")
        plt.xlabel("Tokens (whitespace)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"hist_out_len_{ds}.png"), dpi=200)
        plt.close()

def plot_taxonomy_bars(tax_df: pd.DataFrame, out_dir: str) -> None:
    _ensure_dir(out_dir)
    for ds, g in tax_df.groupby("dataset"):
        g = g.sort_values("count", ascending=False)
        plt.figure()
        plt.bar(g["category"], g["count"])
        plt.title(f"Instruction taxonomy — {ds}")
        plt.xlabel("Category")
        plt.ylabel("Count")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"taxonomy_{ds}.png"), dpi=200)
        plt.close()


def plot_cluster_size_distribution(assignments: pd.DataFrame, out_dir: str, k: int) -> None:
    _ensure_dir(out_dir)
    for ds, g in assignments.groupby("dataset"):
        counts = g["cluster_id"].value_counts().sort_index()

        plt.figure()
        plt.bar(counts.index.astype(int), counts.values)
        plt.title(f"Cluster size distribution — {ds} (k={k})")
        plt.xlabel("Cluster ID")
        plt.ylabel("Examples")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"clusters_{ds}_k{k}.png"), dpi=200)
        plt.close()