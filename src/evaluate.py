from __future__ import annotations

from typing import Dict, List
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.config import TARGET_COLS


def evaluate_models(
    models: Dict[str, object],
    df: pd.DataFrame,
    split_name: str,
    input_col: str,
    model_type: str,
    input_variant: str,
) -> pd.DataFrame:
    rows: List[dict] = []
    X = df[input_col]

    for target in TARGET_COLS:
        model = models[target]
        y_true = df[target]
        y_pred = model.predict(X)

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


def make_prediction_table(
    models: Dict[str, object],
    df: pd.DataFrame,
    input_col: str,
) -> pd.DataFrame:
    out = df.copy()
    X = df[input_col]

    for target in TARGET_COLS:
        pred_col = f"pred_{target.replace(' ', '_')}"
        out[pred_col] = models[target].predict(X)

    return out


def print_classification_reports(
    models: Dict[str, object],
    df: pd.DataFrame,
    split_name: str,
    input_col: str,
) -> None:
    print(f"\n===== Classification Reports: {split_name} =====")
    X = df[input_col]

    for target in TARGET_COLS:
        print(f"\n--- Target: {target} ---")
        y_true = df[target]
        y_pred = models[target].predict(X)
        print(classification_report(y_true, y_pred))