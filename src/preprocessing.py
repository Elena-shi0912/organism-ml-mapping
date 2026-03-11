from __future__ import annotations

import re
import pandas as pd

from src.config import INPUT_CODE_COL, INPUT_NAME_COL, TARGET_COLS


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_and_clean_clsi(csv_path: str | bytes | object) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", na=False)]
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = [INPUT_CODE_COL, INPUT_NAME_COL] + TARGET_COLS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required_cols].copy()

    for col in required_cols:
        df[col] = df[col].apply(normalize_text)

    # Keep rows with at least one usable input
    df = df[
        (df[INPUT_CODE_COL] != "") | (df[INPUT_NAME_COL] != "")
    ].copy()

    # Keep rows with all labels present for now
    for col in TARGET_COLS:
        df = df[df[col] != ""]

    df = df.drop_duplicates().reset_index(drop=True)
    df["row_id"] = range(len(df))

    # Separate input variants for ablations
    df["input_code"] = "species_code=" + df[INPUT_CODE_COL]
    df["input_name"] = "organism=" + df[INPUT_NAME_COL]
    df["input_both"] = df["input_code"] + " " + df["input_name"]

    # Lowercase versions if needed later
    df["input_code_lower"] = df["input_code"].str.lower()
    df["input_name_lower"] = df["input_name"].str.lower()
    df["input_both_lower"] = df["input_both"].str.lower()

    return df