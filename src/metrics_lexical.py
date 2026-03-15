from __future__ import annotations
from collections import Counter
import math
import pandas as pd
from .text_utils import whitespace_tokenize

def shannon_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counter.values():
        p = c / total
        h -= p * math.log(p + 1e-12, 2)
    return float(h)

def corpus_token_counter(texts) -> Counter:
    ctr = Counter()
    for t in texts:
        ctr.update(whitespace_tokenize(t))
    return ctr

def compute_lexical_metrics(df: pd.DataFrame, field: str = "instruction") -> pd.DataFrame:
    rows = []
    for ds, g in df.groupby("dataset"):
        if field == "both":
            texts = (g["instruction"].astype(str) + " " + g["output"].astype(str)).tolist()
        else:
            texts = g[field].astype(str).tolist()

        ctr = corpus_token_counter(texts)
        total_tokens = sum(ctr.values())
        vocab_size = len(ctr)
        ttr = (vocab_size / total_tokens) if total_tokens else 0.0
        ent = shannon_entropy(ctr)

        rows.append({
            "dataset": ds,
            "field": field,
            "total_tokens": int(total_tokens),
            "vocab_size": int(vocab_size),
            "ttr": float(ttr),
            "token_entropy_bits": float(ent),
        })
    return pd.DataFrame(rows).sort_values("dataset")
