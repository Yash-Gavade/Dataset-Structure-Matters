from __future__ import annotations
import pandas as pd

def size_match_sample(df: pd.DataFrame, n_target: int, seed: int) -> pd.DataFrame:
    parts = []
    for name, g in df.groupby("dataset"):
        if len(g) <= n_target:
            parts.append(g.copy())
        else:
            parts.append(g.sample(n=n_target, random_state=seed).copy())
    return pd.concat(parts, ignore_index=True)
