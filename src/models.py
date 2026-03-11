from __future__ import annotations

from typing import Dict
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.config import MODELS_DIR, TARGET_COLS
from src.data_utils import ensure_dir


def build_logreg_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5))),
        ("clf", LogisticRegression(max_iter=3000)),
    ])


def build_rf_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), max_features=5000)),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ])


MODEL_BUILDERS = {
    "logreg": build_logreg_pipeline,
    "random_forest": build_rf_pipeline,
}


def train_models(
    train_df: pd.DataFrame,
    model_type: str = "logreg",
    input_col: str = "input_both",
) -> Dict[str, Pipeline]:
    if model_type not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model_type: {model_type}")

    ensure_dir(MODELS_DIR)
    models: Dict[str, Pipeline] = {}

    X = train_df[input_col]

    for target in TARGET_COLS:
        y = train_df[target]
        model = MODEL_BUILDERS[model_type]()
        model.fit(X, y)
        models[target] = model

    return models


def save_models(models: Dict[str, Pipeline], model_type: str, input_variant: str) -> None:
    ensure_dir(MODELS_DIR)
    for target, model in models.items():
        safe_target = target.replace(" ", "_")
        joblib.dump(model, MODELS_DIR / f"{model_type}_{input_variant}_{safe_target}.joblib")