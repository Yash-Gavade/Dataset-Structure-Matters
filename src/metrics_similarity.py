import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compute_dataset_similarity(df: pd.DataFrame, max_features: int = 50000) -> pd.DataFrame:
    names = []
    texts = []
    for ds, g in df.groupby("dataset"):
        names.append(ds)
        texts.append(" ".join(g["instruction"].astype(str).tolist()))

    vec = TfidfVectorizer(max_features=max_features, stop_words="english", ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    sim = cosine_similarity(X)

    return pd.DataFrame(sim, index=names, columns=names)
