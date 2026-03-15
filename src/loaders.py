from __future__ import annotations
from dataclasses import dataclass
from typing import List

import pandas as pd
from datasets import load_dataset
from .text_utils import normalize_text

@dataclass
class UnifiedExample:
    dataset: str
    instruction: str
    input: str
    output: str

def _to_df(examples: List[UnifiedExample]) -> pd.DataFrame:
    return pd.DataFrame([e.__dict__ for e in examples])

def load_alpaca(split: str = "train") -> pd.DataFrame:
    ds = load_dataset("tatsu-lab/alpaca", split=split)
    ex = []
    for r in ds:
        inst = normalize_text(r.get("instruction", ""))
        inp = normalize_text(r.get("input", ""))
        out = normalize_text(r.get("output", ""))
        if inst and out:
            ex.append(UnifiedExample("alpaca", inst, inp, out))
    return _to_df(ex)

def load_dolly(split: str = "train") -> pd.DataFrame:
    ds = load_dataset("databricks/databricks-dolly-15k", split=split)
    ex = []
    for r in ds:
        inst = normalize_text(r.get("instruction", ""))
        inp = normalize_text(r.get("context", ""))
        out = normalize_text(r.get("response", ""))
        if inst and out:
            ex.append(UnifiedExample("dolly", inst, inp, out))
    return _to_df(ex)

def load_oasst1(split: str = "train", language: str = "en") -> pd.DataFrame:
    '''Flatten OASST1 message trees to single-turn (prompter -> best assistant child) pairs.'''
    ds = load_dataset("OpenAssistant/oasst1", split=split)
    df = ds.to_pandas()

    expected = ["message_id", "parent_id", "role", "text", "lang", "rank"]
    aliases = {"text": ["content"], "lang": ["language"], "rank": ["score"]}
    for col in expected:
        if col in df.columns:
            continue
        found = None
        for a in aliases.get(col, []):
            if a in df.columns:
                found = a
                break
        if found:
            df[col] = df[found]
        else:
            if col in ["rank", "parent_id", "lang"]:
                df[col] = None
            else:
                raise ValueError(f"Expected column '{col}' not found. Columns={list(df.columns)[:30]}...")

    if language:
        df = df[df["lang"] == language].copy()

    df["text_norm"] = df["text"].map(normalize_text)
    df = df[df["text_norm"].str.len() > 0].copy()

    prompters = df[df["role"] == "prompter"][["message_id", "text_norm"]].copy()
    children = df[df["role"] == "assistant"][["message_id", "parent_id", "text_norm", "rank"]].copy()

    children["rank_f"] = children["rank"].fillna(0).astype(float)
    children = children.sort_values(["parent_id", "rank_f", "message_id"], ascending=[True, False, True])
    best = children.groupby("parent_id", as_index=False).first()

    merged = prompters.merge(best, left_on="message_id", right_on="parent_id", how="inner", suffixes=("_p", "_a"))
    out = pd.DataFrame({
        "dataset": "oasst1",
        "instruction": merged["text_norm_p"].values,
        "input": [""] * len(merged),
        "output": merged["text_norm_a"].values,
    })
    return out

def load_all(language: str = "en") -> pd.DataFrame:
    alp = load_alpaca()
    dol = load_dolly()
    oas = load_oasst1(language=language)
    return pd.concat([alp, dol, oas], ignore_index=True)
