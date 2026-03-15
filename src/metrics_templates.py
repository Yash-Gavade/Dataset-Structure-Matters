import pandas as pd
from collections import Counter

def compute_template_openings(df: pd.DataFrame, top_k: int = 20, n_words: int = 3) -> pd.DataFrame:
    rows = []
    for ds, g in df.groupby("dataset"):
        openings = []
        for text in g["instruction"].astype(str).tolist():
            toks = text.split()
            if len(toks) == 0:
                continue
            openings.append(" ".join(toks[:n_words]).lower())

        ctr = Counter(openings)
        total = sum(ctr.values()) if ctr else 0
        top = ctr.most_common(top_k)
        concentration = (sum(c for _, c in top) / total) if total else 0.0

        rows.append({
            "dataset": ds,
            "top_k": int(top_k),
            "n_words": int(n_words),
            "template_concentration": float(concentration),
        })
    return pd.DataFrame(rows).sort_values("dataset")

def top_openings_table(df: pd.DataFrame, top_k: int = 20, n_words: int = 3) -> pd.DataFrame:
    out_rows = []
    for ds, g in df.groupby("dataset"):
        openings = []
        for text in g["instruction"].astype(str).tolist():
            toks = text.split()
            if len(toks) == 0:
                continue
            openings.append(" ".join(toks[:n_words]).lower())
        ctr = Counter(openings)
        for opening, c in ctr.most_common(top_k):
            out_rows.append({"dataset": ds, "opening": opening, "count": int(c), "top_k": int(top_k), "n_words": int(n_words)})
    return pd.DataFrame(out_rows)
