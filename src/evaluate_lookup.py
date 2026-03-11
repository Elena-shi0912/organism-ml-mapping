from __future__ import annotations

from typing import List
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.config import TARGET_COLS
from src.lookup_baselines import LookupModels, get_input_fields


def evaluate_lookup_models(
    models: LookupModels,
    df: pd.DataFrame,
    split_name: str,
    model_type: str,
    input_variant: str,
) -> pd.DataFrame:
    rows: List[dict] = []
    X = get_input_fields(df, input_variant)

    for target in TARGET_COLS:
        y_true = df[target]
        y_pred = models.predict_target(target, X)

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


def print_lookup_classification_reports(
    models: LookupModels,
    df: pd.DataFrame,
    split_name: str,
    input_variant: str,
) -> None:
    print(f"\n===== Lookup Classification Reports: {split_name} =====")
    X = get_input_fields(df, input_variant)

    for target in TARGET_COLS:
        print(f"\n--- Target: {target} ---")
        y_true = df[target]
        y_pred = models.predict_target(target, X)
        print(classification_report(y_true, y_pred))