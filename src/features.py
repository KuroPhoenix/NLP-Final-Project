from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def mcq_flag(q: pd.Series) -> pd.Series:
    q = q.fillna("").astype(str)
    has_choices = q.str.contains("Answer Choices", case=False, regex=False)
    lettered = q.str.contains(r"\n\s*[A-E][\.\)]\s", regex=True)
    return has_choices | lettered


def handcrafted_features(queries: pd.Series) -> pd.DataFrame:
    q = queries.fillna("").astype(str)
    f = pd.DataFrame(index=q.index)
    f["char_len"] = q.str.len().astype(np.float32)
    f["word_count"] = q.str.split().str.len().fillna(0).astype(np.float32)
    f["log_char_len"] = np.log1p(f["char_len"])
    f["log_word_count"] = np.log1p(f["word_count"])
    f["line_count"] = (q.str.count("\n") + 1).astype(np.float32)
    f["digit_ratio"] = (q.str.count(r"\d") / (f["char_len"] + 1.0)).astype(np.float32)
    f["latex"] = q.str.count(r"\$|\\frac|\\sqrt|\\sum|\\angle").astype(np.float32)
    f["code"] = q.str.count(r"```|def |class |#include|import |SELECT |function ").astype(np.float32)
    f["question_marks"] = q.str.count(r"\?").astype(np.float32)
    f["is_mcq"] = mcq_flag(q).astype(np.float32)
    f["n_choices"] = q.str.count(r"\n\s*[A-E][\.\)]\s").astype(np.float32)
    f["is_long"] = (f["char_len"] > 5000).astype(np.float32)
    f["is_short"] = (f["char_len"] < 300).astype(np.float32)
    return f.astype(np.float32)


def handcrafted_matrix(queries: pd.Series) -> np.ndarray:
    return handcrafted_features(queries).to_numpy(np.float32)


def cap_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.65)
    tail = max_chars - head
    return text[:head] + "\n[...]\n" + text[-tail:]


def build_tfidf(cfg) -> FeatureUnion:
    word = TfidfVectorizer(lowercase=True, strip_accents="unicode", sublinear_tf=True,
                           ngram_range=(1, 2), min_df=2, max_features=cfg.tfidf_word_features)
    char = TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5),
                           min_df=2, max_features=cfg.tfidf_char_features)
    return FeatureUnion([("word", word), ("char", char)])


def _embed_cache_path(cfg, split: str, n: int) -> Path:
    return cfg.cache_dir / f"{cfg.embedding_cache_name}_{split}_{n}.npy"


def load_or_compute_embeddings(cfg, df: pd.DataFrame, split: str) -> np.ndarray:
    """Reuse cached Qwen3 embeddings if present (search any Output/*/cache); else compute (GPU)."""
    n = len(df)
    target = _embed_cache_path(cfg, split, n)
    if target.exists():
        return np.load(target)
    fname = f"{cfg.embedding_cache_name}_{split}_{n}.npy"
    candidates = [cfg.prev_cache_dir / fname,
                  cfg.root / "Output" / "router_a100_exact_metric_v2" / "cache" / fname]
    candidates += sorted((cfg.root / "Output").rglob(fname))
    for c in candidates:
        if c.exists():
            emb = np.load(c)
            np.save(target, emb)
            return emb
    return _compute_embeddings(cfg, df, split, target)


def _compute_embeddings(cfg, df, split, target) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    import torch
    kwargs = dict(model_name_or_path=cfg.qwen_model_id, truncate_dim=cfg.embedding_dim,
                  trust_remote_code=True)
    if torch.cuda.is_available():
        model = SentenceTransformer(**kwargs,
                                    model_kwargs={"torch_dtype": torch.float16, "device_map": "auto"},
                                    tokenizer_kwargs={"padding_side": "left"})
    else:
        model = SentenceTransformer(**kwargs)
    texts = [cap_text(t, 12000) for t in df["query"].fillna("").astype(str).tolist()]
    emb = model.encode(texts, batch_size=8, show_progress_bar=True,
                       normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    np.save(target, emb)
    return emb
