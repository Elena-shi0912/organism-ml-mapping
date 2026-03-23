from __future__ import annotations

from typing import Iterable, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TEST_SIZE, VAL_SIZE, STRATIFY_MIN_COUNT


def make_stratify_label(
    df: pd.DataFrame,
    target_col: str,
    min_count: int = STRATIFY_MIN_COUNT,
) -> pd.Series:
    labels = df[target_col].copy()
    counts = labels.value_counts()
    rare = counts[counts < min_count].index
    labels = labels.where(~labels.isin(rare), "__RARE__")
    return labels


def make_stratify_label_multi(
    df: pd.DataFrame,
    target_cols: Iterable[str],
    min_count: int = STRATIFY_MIN_COUNT,
) -> pd.Series:
    cols = list(target_cols)
    if not cols:
        raise ValueError("target_cols must be non-empty")
    if len(cols) == 1:
        return make_stratify_label(df, cols[0], min_count=min_count)
    combo = df[cols].astype(str).agg("||".join, axis=1)
    temp = pd.DataFrame({"combo": combo})
    temp["combo"] = make_stratify_label(temp, "combo", min_count=min_count)
    return temp["combo"]


def split_train_val_test(
    df: pd.DataFrame,
    stratify_col: str | Iterable[str] = "CLSI Breakpoint Group",
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be < 1")

    if isinstance(stratify_col, str):
        stratify_all = make_stratify_label(df, stratify_col)
    else:
        stratify_all = make_stratify_label_multi(df, stratify_col)

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_all,
    )

    val_relative_size = val_size / (1 - test_size)
    if isinstance(stratify_col, str):
        stratify_train_val = make_stratify_label(train_val_df, stratify_col)
    else:
        stratify_train_val = make_stratify_label_multi(train_val_df, stratify_col)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        random_state=random_state,
        stratify=stratify_train_val,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
