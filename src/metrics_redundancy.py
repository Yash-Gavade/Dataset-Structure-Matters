from __future__ import annotations
from collections import Counter
import hashlib
import pandas as pd
from .text_utils import safe_lower, whitespace_tokenize

def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def duplicate_rate(df: pd.DataFrame, field: str = "instruction") -> pd.DataFrame:
    rows = []
    for ds, g in df.groupby("dataset"):
        vals = g[field].astype(str).map(safe_lower).tolist()
        hashes = [_hash(v) for v in vals]
        ctr = Counter(hashes)
        n = len(hashes)
        dup = sum(c - 1 for c in ctr.values() if c > 1)
        rate = dup / n if n else 0.0
        rows.append({"dataset": ds, "field": field, "n": n, "n_duplicate_rows": int(dup), "duplicate_rate": float(rate)})
    return pd.DataFrame(rows).sort_values("dataset")

def top_ngrams(df: pd.DataFrame, field: str = "instruction", n: int = 2, topk: int = 50) -> pd.DataFrame:
    rows = []
    for ds, g in df.groupby("dataset"):
        ctr = Counter()
        for t in g[field].astype(str).tolist():
            toks = [tok.lower() for tok in whitespace_tokenize(t)]
            for i in range(len(toks) - n + 1):
                ctr[tuple(toks[i:i+n])] += 1
        for gram, c in ctr.most_common(topk):
            rows.append({"dataset": ds, "field": field, "n": n, "ngram": " ".join(gram), "count": int(c)})
    return pd.DataFrame(rows)
