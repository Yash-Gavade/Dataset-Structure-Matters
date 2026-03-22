from __future__ import annotations
import argparse
import json
import os
from datetime import datetime

import pandas as pd

from src.loaders import load_all
from src.sampling import size_match_sample
from src.metrics_lengths import compute_length_metrics
from src.metrics_lexical import compute_lexical_metrics
from src.metrics_redundancy import duplicate_rate, top_ngrams
from src.clustering import cluster_instructions
from src.visualize import plot_length_histograms, plot_taxonomy_bars

from src.metrics_bootstrap import length_mean_ci
from src.metrics_distribution_shape import compute_distribution_shape
from src.metrics_taxonomy import compute_taxonomy
from src.metrics_similarity import compute_dataset_similarity
from src.metrics_templates import compute_template_openings, top_openings_table

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_target", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--language", type=str, default="en")
    ap.add_argument("--out_dir", type=str, default="outputs_extended")
    ap.add_argument("--k_values", type=int, nargs="+", default=[50, 75, 100])
    ap.add_argument("--max_features", type=int, default=50000)
    ap.add_argument("--min_df", type=int, default=2)
    ap.add_argument("--ngram_topk", type=int, default=50)
    ap.add_argument("--bootstrap_n", type=int, default=1000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    ngram_dir = os.path.join(args.out_dir, "top_ngrams")
    os.makedirs(ngram_dir, exist_ok=True)

    # 1) Load + unify
    df = load_all(language=args.language)

    # 2) Size-match for comparability
    df_sm = size_match_sample(df, n_target=args.n_target, seed=args.seed)

    # 3) Baseline metrics
    length_df = compute_length_metrics(df_sm)
    lex_instr = compute_lexical_metrics(df_sm, field="instruction")
    lex_out = compute_lexical_metrics(df_sm, field="output")

    dup_instr = duplicate_rate(df_sm, field="instruction")
    dup_out = duplicate_rate(df_sm, field="output")

    # N-grams
    for n in [2, 3]:
        ng = top_ngrams(df_sm, field="instruction", n=n, topk=args.ngram_topk)
        ng.to_csv(os.path.join(ngram_dir, f"top_{n}gram_instruction.csv"), index=False)

    # 4) Extended metrics
    ci_df = length_mean_ci(df_sm, n_boot=args.bootstrap_n, ci=0.95, seed=args.seed)
    shape_df = compute_distribution_shape(df_sm)
    tax_df = compute_taxonomy(df_sm)
    tmpl_df = compute_template_openings(df_sm, top_k=20, n_words=3)
    top_open_df = top_openings_table(df_sm, top_k=20, n_words=3)
    sim_df = compute_dataset_similarity(df_sm, max_features=args.max_features)

    # 5) Clustering stability across multiple k
    cluster_metrics_all = []
    for k in args.k_values:
        cm, assignments = cluster_instructions(
            df_sm,
            k=k,
            max_features=args.max_features,
            min_df=args.min_df,
            seed=args.seed,
        )
        cm["k"] = k
        cluster_metrics_all.append(cm)

        # plot cluster size distribution per dataset (one plot per dataset per k)
        for ds, g in assignments.groupby("dataset"):
            counts = g["cluster_id"].value_counts().sort_index()
            import matplotlib.pyplot as plt
            plt.figure()
            plt.bar(counts.index.astype(int), counts.values)
            plt.title(f"Cluster size distribution — {ds} (k={k})")
            plt.xlabel("Cluster ID")
            plt.ylabel("Examples")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"clusters_{ds}_k{k}.png"), dpi=200)
            plt.close()

    cluster_metrics_by_k = pd.concat(cluster_metrics_all, ignore_index=True)

    # 6) Save tables
    length_df.to_csv(os.path.join(args.out_dir, "length_metrics.csv"), index=False)
    lex_instr.to_csv(os.path.join(args.out_dir, "lexical_instruction.csv"), index=False)
    lex_out.to_csv(os.path.join(args.out_dir, "lexical_output.csv"), index=False)
    dup_instr.to_csv(os.path.join(args.out_dir, "dup_instruction.csv"), index=False)
    dup_out.to_csv(os.path.join(args.out_dir, "dup_output.csv"), index=False)

    ci_df.to_csv(os.path.join(args.out_dir, "length_ci.csv"), index=False)
    shape_df.to_csv(os.path.join(args.out_dir, "distribution_shape.csv"), index=False)
    tax_df.to_csv(os.path.join(args.out_dir, "taxonomy_counts.csv"), index=False)
    tmpl_df.to_csv(os.path.join(args.out_dir, "template_concentration.csv"), index=False)
    top_open_df.to_csv(os.path.join(args.out_dir, "top_openings.csv"), index=False)
    sim_df.to_csv(os.path.join(args.out_dir, "dataset_similarity.csv"))

    cluster_metrics_by_k.to_csv(os.path.join(args.out_dir, "cluster_metrics_by_k.csv"), index=False)

    # 7) Plots
    plot_length_histograms(df_sm, fig_dir)
    plot_taxonomy_bars(tax_df, fig_dir)

    # 8) One combined summary table (easy to paste into paper)
    summary = length_df.merge(
        dup_instr[["dataset", "duplicate_rate"]].rename(columns={"duplicate_rate": "dup_rate_instruction"}),
        on="dataset",
    ).merge(
        dup_out[["dataset", "duplicate_rate"]].rename(columns={"duplicate_rate": "dup_rate_output"}),
        on="dataset",
    ).merge(
        lex_instr[["dataset", "vocab_size", "ttr", "token_entropy_bits"]].rename(
            columns={"vocab_size": "instr_vocab", "ttr": "instr_ttr", "token_entropy_bits": "instr_entropy_bits"}
        ),
        on="dataset",
    ).merge(
        lex_out[["dataset", "vocab_size", "ttr", "token_entropy_bits"]].rename(
            columns={"vocab_size": "out_vocab", "ttr": "out_ttr", "token_entropy_bits": "out_entropy_bits"}
        ),
        on="dataset",
    ).merge(
        ci_df[["dataset", "instr_mean_ci_low", "instr_mean_ci_high", "out_mean_ci_low", "out_mean_ci_high"]],
        on="dataset",
    ).merge(
        shape_df[["dataset", "instr_std", "instr_skew", "instr_kurtosis", "out_std", "out_skew", "out_kurtosis"]],
        on="dataset",
    ).merge(
        tmpl_df[["dataset", "template_concentration"]],
        on="dataset",
    )

    summary.to_csv(os.path.join(args.out_dir, "summary_metrics_extended.csv"), index=False)

    # 9) Metadata for reproducibility
    meta = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "args": vars(args),
        "n_loaded": int(len(df)),
        "n_after_size_match": int(len(df_sm)),
        "n_by_dataset": df_sm["dataset"].value_counts().to_dict(),
    }
    with open(os.path.join(args.out_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Done. Outputs in:", args.out_dir)

if __name__ == "__main__":
    main()
