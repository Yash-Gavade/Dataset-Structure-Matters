from __future__ import annotations
import argparse
import json
import os
from datetime import datetime

from src.config import DEFAULTS
from src.loaders import load_all
from src.sampling import size_match_sample
from src.metrics_lengths import compute_length_metrics
from src.metrics_lexical import compute_lexical_metrics
from src.metrics_redundancy import duplicate_rate, top_ngrams
from src.clustering import cluster_instructions
from src.visualize import plot_length_histograms, plot_cluster_size_distribution

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_target", type=int, default=DEFAULTS["n_target"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--k", type=int, default=DEFAULTS["k"])
    ap.add_argument("--language", type=str, default=DEFAULTS["language"])
    ap.add_argument("--out_dir", type=str, default="outputs")
    ap.add_argument("--max_features", type=int, default=DEFAULTS["max_features"])
    ap.add_argument("--min_df", type=int, default=DEFAULTS["min_df"])
    ap.add_argument("--ngram_topk", type=int, default=DEFAULTS["ngram_topk"])
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

    # 3) Metrics
    length_df = compute_length_metrics(df_sm)
    lex_instr = compute_lexical_metrics(df_sm, field="instruction")
    lex_out = compute_lexical_metrics(df_sm, field="output")

    dup_instr = duplicate_rate(df_sm, field="instruction")
    dup_out = duplicate_rate(df_sm, field="output")

    # 4) N-grams
    for n in [2, 3]:
        ng = top_ngrams(df_sm, field="instruction", n=n, topk=args.ngram_topk)
        ng.to_csv(os.path.join(ngram_dir, f"top_{n}gram_instruction.csv"), index=False)

    # 5) Clustering
    cluster_metrics, assignments = cluster_instructions(
        df_sm,
        k=args.k,
        max_features=args.max_features,
        min_df=args.min_df,
        seed=args.seed,
    )

    # 6) Save tables
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
        cluster_metrics[["dataset", "cluster_entropy_bits", "cluster_entropy_norm"]],
        on="dataset",
    )

    summary.to_csv(os.path.join(args.out_dir, "summary_metrics.csv"), index=False)
    cluster_metrics.to_csv(os.path.join(args.out_dir, "cluster_metrics.csv"), index=False)

    # 7) Plots
    plot_length_histograms(df_sm, fig_dir)
    plot_cluster_size_distribution(assignments, fig_dir, k=args.k)

    # 8) Metadata for reproducibility
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
