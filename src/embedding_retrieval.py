from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize as l2_normalize

from src.config import TARGET_COLS
from src.lookup_baselines import get_input_fields, normalize_for_lookup


@dataclass
class LsaIndex:
    input_variant: str
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD
    matrix: np.ndarray
    labels: Dict[str, List[str]]
    normalize: bool = True


def _normalize_series(series: pd.Series) -> pd.Series:
    return series.apply(normalize_for_lookup)


def _fit_lsa_components(tfidf_matrix, max_components: int = 300) -> TruncatedSVD:
    n_features = tfidf_matrix.shape[1]
    n_components = max(2, min(max_components, n_features - 1))
    return TruncatedSVD(n_components=n_components, random_state=42)


def build_lsa_index(
    index_df: pd.DataFrame,
    input_variant: str,
    normalize: bool = True,
) -> LsaIndex:
    X = get_input_fields(index_df, input_variant)
    if normalize:
        X = _normalize_series(X)

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5))
    tfidf_matrix = vectorizer.fit_transform(X)

    svd = _fit_lsa_components(tfidf_matrix)
    dense = svd.fit_transform(tfidf_matrix)
    dense = l2_normalize(dense, axis=1)

    labels = {target: index_df[target].astype(str).tolist() for target in TARGET_COLS}

    return LsaIndex(
        input_variant=input_variant,
        vectorizer=vectorizer,
        svd=svd,
        matrix=dense,
        labels=labels,
        normalize=normalize,
    )


def _majority_vote(candidates: List[str], top1_label: str) -> str:
    counts: Dict[str, int] = {}
    for item in candidates:
        counts[item] = counts.get(item, 0) + 1
    max_count = max(counts.values())
    tied = [label for label, count in counts.items() if count == max_count]
    return top1_label if top1_label in tied else tied[0]


def retrieve_lsa_topk(
    index: LsaIndex,
    df: pd.DataFrame,
    k: int = 3,
) -> Dict[str, List[List[tuple[str, float]]]]:
    X = get_input_fields(df, index.input_variant)
    if index.normalize:
        X = _normalize_series(X)

    query_tfidf = index.vectorizer.transform(X)
    query_dense = index.svd.transform(query_tfidf)
    query_dense = l2_normalize(query_dense, axis=1)

    similarities = query_dense @ index.matrix.T
    topk_idx = np.argsort(-similarities, axis=1)[:, :k]
    topk_scores = np.take_along_axis(similarities, topk_idx, axis=1)

    candidates: Dict[str, List[List[tuple[str, float]]]] = {target: [] for target in TARGET_COLS}
    for row_idx in range(topk_idx.shape[0]):
        indices = topk_idx[row_idx]
        scores = topk_scores[row_idx]
        for target in TARGET_COLS:
            row_candidates = [
                (index.labels[target][i], float(score))
                for i, score in zip(indices, scores)
            ]
            candidates[target].append(row_candidates)

    return candidates


def predict_lsa_retrieval(
    index: LsaIndex,
    df: pd.DataFrame,
    k: int = 1,
) -> Dict[str, List[str]]:
    candidates = retrieve_lsa_topk(index, df, k=max(3, k))

    preds: Dict[str, List[str]] = {target: [] for target in TARGET_COLS}
    num_rows = len(df)
    for row_idx in range(num_rows):
        for target in TARGET_COLS:
            row_candidates = candidates[target][row_idx]
            top1_label = row_candidates[0][0]
            labels_only = [label for label, _ in row_candidates[:k]]
            pred = top1_label if k == 1 else _majority_vote(labels_only, top1_label)
            preds[target].append(pred)

    return preds


def make_lsa_prediction_table(
    index: LsaIndex,
    df: pd.DataFrame,
    k: int = 1,
) -> pd.DataFrame:
    out = df.copy()
    preds = predict_lsa_retrieval(index, df, k=k)
    for target in TARGET_COLS:
        pred_col = f"pred_{target.replace(' ', '_')}"
        out[pred_col] = preds[target]
    return out


def evaluate_lsa_retrieval(
    index: LsaIndex,
    df: pd.DataFrame,
    split_name: str,
    input_variant: str,
    model_type: str,
    k: int = 1,
) -> pd.DataFrame:
    preds = predict_lsa_retrieval(index, df, k=k)
    rows = []
    for target in TARGET_COLS:
        y_true = df[target]
        y_pred = preds[target]
        rows.append({
            "model_type": model_type,
            "input_variant": input_variant,
            "split": split_name,
            "target": target,
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        })
    return pd.DataFrame(rows)
