from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd
from .text_utils import whitespace_tokenize

def _summary(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
    }

def compute_length_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, g in df.groupby("dataset"):
        inst_l = np.array([len(whitespace_tokenize(x)) for x in g["instruction"].tolist()], dtype=float)
        out_l = np.array([len(whitespace_tokenize(x)) for x in g["output"].tolist()], dtype=float)
        inst_s = _summary(inst_l)
        out_s = _summary(out_l)
        ratio = float(np.mean((out_l + 1.0) / (inst_l + 1.0))) if len(g) else 0.0
        rows.append({
            "dataset": ds,
            "n_examples": int(len(g)),
            "instr_mean": inst_s["mean"],
            "instr_median": inst_s["median"],
            "instr_p90": inst_s["p90"],
            "out_mean": out_s["mean"],
            "out_median": out_s["median"],
            "out_p90": out_s["p90"],
            "out_to_instr_ratio_mean": ratio,
        })
    return pd.DataFrame(rows).sort_values("dataset")
