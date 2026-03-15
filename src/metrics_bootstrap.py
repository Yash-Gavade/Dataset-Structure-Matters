import numpy as np
import pandas as pd

def bootstrap_ci(values, n_boot=1000, ci=0.95, seed=42, stat="mean"):
    rng = np.random.default_rng(seed)
    values = np.array(values, dtype=float)
    if len(values) == 0:
        return 0.0, 0.0

    stats = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        if stat == "mean":
            stats.append(np.mean(sample))
        elif stat == "median":
            stats.append(np.median(sample))
        else:
            raise ValueError("stat must be 'mean' or 'median'")

    lower = np.percentile(stats, (1 - ci) / 2 * 100)
    upper = np.percentile(stats, (1 + ci) / 2 * 100)
    return float(lower), float(upper)

def length_mean_ci(df: pd.DataFrame, n_boot=1000, ci=0.95, seed=42) -> pd.DataFrame:
    rows = []
    for ds, g in df.groupby("dataset"):
        inst = g["instruction"].astype(str).str.split().map(len).values
        out = g["output"].astype(str).str.split().map(len).values
        inst_ci = bootstrap_ci(inst, n_boot=n_boot, ci=ci, seed=seed, stat="mean")
        out_ci = bootstrap_ci(out, n_boot=n_boot, ci=ci, seed=seed, stat="mean")
        rows.append({
            "dataset": ds,
            "instr_mean_ci_low": inst_ci[0],
            "instr_mean_ci_high": inst_ci[1],
            "out_mean_ci_low": out_ci[0],
            "out_mean_ci_high": out_ci[1],
            "n_boot": int(n_boot),
            "ci_level": float(ci),
        })
    return pd.DataFrame(rows).sort_values("dataset")
