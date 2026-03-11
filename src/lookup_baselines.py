from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from src.config import INPUT_CODE_COL, INPUT_NAME_COL, TARGET_COLS


def normalize_for_lookup(text: object) -> str:
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"[.,;:()\-_/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_input_fields(df: pd.DataFrame, input_variant: str) -> pd.Series:
    """
    Build lookup keys in exactly the same format as the ML input variants.
    """
    code = df[INPUT_CODE_COL].fillna("").astype(str)
    name = df[INPUT_NAME_COL].fillna("").astype(str)

    if input_variant == "code_only":
        return "species_code=" + code

    if input_variant == "name_only":
        return "organism=" + name

    if input_variant == "code_plus_name":
        return "species_code=" + code + " organism=" + name

    raise ValueError(f"Unknown input_variant: {input_variant}")


@dataclass
class LookupModels:
    mode: str
    input_variant: str
    label_maps: Dict[str, Dict[str, str]]

    def predict_target(self, target: str, X: pd.Series) -> List[str]:
        label_map = self.label_maps[target]
        preds = []
        for x in X:
            key = x if self.mode == "exact" else normalize_for_lookup(x)
            preds.append(label_map.get(key, "__UNKNOWN__"))
        return preds


def build_lookup_models_from_full_clsi(
    full_clsi_df: pd.DataFrame,
    input_variant: str = "code_plus_name",
    mode: str = "exact",
) -> LookupModels:
    """
    Build lookup maps from the full cleaned CLSI table, not from train split.
    """
    if mode not in {"exact", "normalized"}:
        raise ValueError(f"Unsupported lookup mode: {mode}")

    X = get_input_fields(full_clsi_df, input_variant)
    label_maps: Dict[str, Dict[str, str]] = {}

    for target in TARGET_COLS:
        y = full_clsi_df[target].astype(str)

        temp = pd.DataFrame({"x": X, "y": y}).copy()
        temp["key"] = temp["x"] if mode == "exact" else temp["x"].apply(normalize_for_lookup)

        # If duplicate keys map to multiple labels, keep the most frequent label.
        grouped = (
            temp.groupby(["key", "y"])
            .size()
            .reset_index(name="count")
            .sort_values(["key", "count"], ascending=[True, False])
        )
        best = grouped.drop_duplicates(subset=["key"], keep="first")

        label_maps[target] = dict(zip(best["key"], best["y"]))

    return LookupModels(mode=mode, input_variant=input_variant, label_maps=label_maps)


def make_lookup_prediction_table(
    models: LookupModels,
    df: pd.DataFrame,
) -> pd.DataFrame:
    out = df.copy()
    X = get_input_fields(df, models.input_variant)

    for target in TARGET_COLS:
        pred_col = f"pred_{target.replace(' ', '_')}"
        out[pred_col] = models.predict_target(target, X)

    return out