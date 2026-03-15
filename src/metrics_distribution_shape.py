import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

def compute_distribution_shape(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, g in df.groupby("dataset"):
        instr_len = g["instruction"].astype(str).str.split().map(len).values.astype(float)
        out_len = g["output"].astype(str).str.split().map(len).values.astype(float)

        rows.append({
            "dataset": ds,
            "instr_std": float(np.std(instr_len)),
            "instr_skew": float(skew(instr_len)) if len(instr_len) else 0.0,
            "instr_kurtosis": float(kurtosis(instr_len)) if len(instr_len) else 0.0,
            "out_std": float(np.std(out_len)),
            "out_skew": float(skew(out_len)) if len(out_len) else 0.0,
            "out_kurtosis": float(kurtosis(out_len)) if len(out_len) else 0.0,
        })
    return pd.DataFrame(rows).sort_values("dataset")
