from __future__ import annotations
import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from .text_utils import safe_lower

def entropy_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def cluster_instructions(df: pd.DataFrame, k: int, max_features: int = 50000, min_df: int = 2, seed: int = 42):
    metrics_rows = []
    assign_parts = []

    for ds, g in df.groupby("dataset"):
        texts = g["instruction"].astype(str).map(safe_lower).tolist()

        vec = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            ngram_range=(1, 2),
            stop_words="english",
        )
        X = vec.fit_transform(texts)

        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X)

        counts = np.bincount(labels, minlength=k)
        ent = entropy_from_counts(counts)
        ent_norm = ent / math.log2(k) if k > 1 else 0.0

        metrics_rows.append({
            "dataset": ds,
            "k": int(k),
            "cluster_entropy_bits": float(ent),
            "cluster_entropy_norm": float(ent_norm),
            "cluster_size_min": int(counts.min()) if len(counts) else 0,
            "cluster_size_max": int(counts.max()) if len(counts) else 0,
        })

        tmp = g[["dataset", "instruction", "output"]].copy()
        tmp["cluster_id"] = labels
        assign_parts.append(tmp)

    return pd.DataFrame(metrics_rows).sort_values(["k","dataset"]), pd.concat(assign_parts, ignore_index=True)
