from __future__ import annotations

import random
import re
from typing import List
import pandas as pd

from src.config import INPUT_CODE_COL, INPUT_NAME_COL, NOISY_COPIES_PER_ROW, RANDOM_STATE

random.seed(RANDOM_STATE)


def collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def maybe_lowercase(text: str) -> str:
    return text.lower() if random.random() < 0.5 else text


def abbreviate_genus(name: str) -> str:
    parts = name.split()
    if len(parts) >= 2:
        return f"{parts[0][0]}. {' '.join(parts[1:])}"
    return name


def remove_genus_period_variant(name: str) -> str:
    parts = name.split()
    if len(parts) >= 2:
        return f"{parts[0][0]} {' '.join(parts[1:])}"
    return name


def drop_first_token(name: str) -> str:
    parts = name.split()
    if len(parts) >= 2:
        return " ".join(parts[1:])
    return name


def simple_typo(text: str) -> str:
    if len(text) < 5:
        return text
    idxs = [i for i, ch in enumerate(text) if ch.isalpha()]
    if not idxs:
        return text
    i = random.choice(idxs)
    op = random.choice(["delete", "swap", "duplicate"])
    chars = list(text)

    if op == "delete" and len(chars) > 1:
        del chars[i]
    elif op == "swap" and i < len(chars) - 1:
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    elif op == "duplicate":
        chars.insert(i, chars[i])

    return "".join(chars)


def perturb_species_name(name: str) -> str:
    candidates = [
        name,
        maybe_lowercase(name),
        abbreviate_genus(name),
        remove_genus_period_variant(name),
        drop_first_token(name),
        simple_typo(name),
    ]
    candidates = [collapse_spaces(c) for c in candidates if c.strip()]
    candidates = list(dict.fromkeys(candidates))
    return random.choice(candidates)


def perturb_species_code(code: str) -> str:
    candidates = [code, code.lower(), code.upper(), simple_typo(code)]
    candidates = [c.strip() for c in candidates if c.strip()]
    candidates = list(dict.fromkeys(candidates))
    return random.choice(candidates)


def refresh_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["input_code"] = "species_code=" + df[INPUT_CODE_COL]
    df["input_name"] = "organism=" + df[INPUT_NAME_COL]
    df["input_both"] = df["input_code"] + " " + df["input_name"]
    df["input_code_lower"] = df["input_code"].str.lower()
    df["input_name_lower"] = df["input_name"].str.lower()
    df["input_both_lower"] = df["input_both"].str.lower()
    return df


def create_noisy_copy(row: pd.Series) -> pd.Series:
    new_row = row.copy()
    new_row[INPUT_NAME_COL] = perturb_species_name(str(row[INPUT_NAME_COL]))
    new_row[INPUT_CODE_COL] = perturb_species_code(str(row[INPUT_CODE_COL]))
    new_row["is_noisy_augmented"] = 1
    return new_row


def augment_training_data(
    train_df: pd.DataFrame,
    noisy_copies_per_row: int = NOISY_COPIES_PER_ROW,
) -> pd.DataFrame:
    base = train_df.copy()
    base["is_noisy_augmented"] = 0

    augmented_rows: List[pd.Series] = []
    for _, row in train_df.iterrows():
        for _ in range(noisy_copies_per_row):
            augmented_rows.append(create_noisy_copy(row))

    noisy_df = pd.DataFrame(augmented_rows)
    combined = pd.concat([base, noisy_df], ignore_index=True)
    combined = refresh_input_columns(combined)
    combined = combined.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    return combined


def create_noisy_test_set(test_df: pd.DataFrame) -> pd.DataFrame:
    noisy_rows = []
    for _, row in test_df.iterrows():
        noisy_rows.append(create_noisy_copy(row))
    noisy_df = pd.DataFrame(noisy_rows).reset_index(drop=True)
    noisy_df["is_noisy_augmented"] = 1
    noisy_df = refresh_input_columns(noisy_df)
    return noisy_df