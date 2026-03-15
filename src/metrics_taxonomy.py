import pandas as pd

def classify_instruction(text: str) -> str:
    t = (text or "").lower()

    # simple, reproducible rules (easy to explain in paper)
    if any(k in t for k in ["summarize", "summary", "tl;dr"]):
        return "summarization"
    if any(k in t for k in ["translate", "translation"]):
        return "translation"
    if any(k in t for k in ["write a story", "poem", "creative", "novel", "character"]):
        return "creative"
    if any(k in t for k in ["explain", "describe", "what is", "definition"]):
        return "explanation"
    if any(k in t for k in ["code", "python", "java", "c++", "bug", "function", "sql"]):
        return "coding"
    if any(k in t for k in ["why", "how", "reason", "justify", "step by step"]):
        return "qa_reasoning"
    if any(k in t for k in ["chat", "roleplay", "act as", "conversation"]):
        return "conversation"
    return "other"

def compute_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["category"] = tmp["instruction"].astype(str).apply(classify_instruction)
    out = tmp.groupby(["dataset", "category"]).size().reset_index(name="count")
    # also add proportions
    totals = out.groupby("dataset")["count"].sum().reset_index(name="total")
    out = out.merge(totals, on="dataset")
    out["proportion"] = out["count"] / out["total"]
    return out.sort_values(["dataset", "count"], ascending=[True, False])
