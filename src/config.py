DATASETS = {
    "alpaca": {"hf_name": "tatsu-lab/alpaca"},
    "dolly": {"hf_name": "databricks/databricks-dolly-15k"},
    "oasst1": {"hf_name": "OpenAssistant/oasst1"},
}

DEFAULTS = {
    "seed": 42,
    "n_target": 10000,
    "k": 75,
    "max_features": 50000,
    "min_df": 2,
    "ngram_topk": 50,
    "language": "en",
}
