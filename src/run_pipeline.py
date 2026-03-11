from __future__ import annotations

import pandas as pd

from src.config import (
    CLSI_PATH,
    PROCESSED_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
)
from src.data_utils import ensure_dir, save_csv
from src.preprocessing import load_and_clean_clsi
from src.split import split_train_val_test
from src.add_noise import augment_training_data, create_noisy_test_set
from src.models import train_models, save_models
from src.evaluate import evaluate_models, make_prediction_table, print_classification_reports
from src.lookup_baselines import build_lookup_models_from_full_clsi, make_lookup_prediction_table
from src.evaluate_lookup import evaluate_lookup_models, print_lookup_classification_reports

INPUT_VARIANTS = {
    "code_only": "input_code",
    "name_only": "input_name",
    "code_plus_name": "input_both",
}

ML_MODEL_TYPES = ["logreg", "random_forest"]
# ML_MODEL_TYPES = []
LOOKUP_MODEL_TYPES = ["exact_lookup", "normalized_lookup"]


def run_lookup_experiments(
    full_clsi_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_clean_df: pd.DataFrame,
    test_noisy_df: pd.DataFrame,
) -> list[pd.DataFrame]:
    all_metrics = []

    for lookup_type in LOOKUP_MODEL_TYPES:
        mode = "exact" if lookup_type == "exact_lookup" else "normalized"

        for input_variant in INPUT_VARIANTS.keys():
            print(f"\nRunning lookup baseline={lookup_type}, input_variant={input_variant}")

            lookup_models = build_lookup_models_from_full_clsi(
                full_clsi_df=full_clsi_df,
                input_variant=input_variant,
                mode=mode,
            )

            val_metrics = evaluate_lookup_models(
                models=lookup_models,
                df=val_df,
                split_name="val_clean",
                model_type=lookup_type,
                input_variant=input_variant,
            )
            test_clean_metrics = evaluate_lookup_models(
                models=lookup_models,
                df=test_clean_df,
                split_name="test_clean",
                model_type=lookup_type,
                input_variant=input_variant,
            )
            test_noisy_metrics = evaluate_lookup_models(
                models=lookup_models,
                df=test_noisy_df,
                split_name="test_noisy",
                model_type=lookup_type,
                input_variant=input_variant,
            )

            metrics_df = pd.concat(
                [val_metrics, test_clean_metrics, test_noisy_metrics],
                ignore_index=True,
            )
            all_metrics.append(metrics_df)

            save_csv(
                metrics_df,
                METRICS_DIR / f"{lookup_type}_{input_variant}_metrics.csv",
            )

            pred_clean = make_lookup_prediction_table(lookup_models, test_clean_df)
            pred_noisy = make_lookup_prediction_table(lookup_models, test_noisy_df)

            save_csv(
                pred_clean,
                PREDICTIONS_DIR / f"{lookup_type}_{input_variant}_test_clean_predictions.csv",
            )
            save_csv(
                pred_noisy,
                PREDICTIONS_DIR / f"{lookup_type}_{input_variant}_test_noisy_predictions.csv",
            )

            print_lookup_classification_reports(
                lookup_models,
                test_clean_df,
                split_name=f"{lookup_type}_{input_variant}_test_clean",
                input_variant=input_variant,
            )
            print_lookup_classification_reports(
                lookup_models,
                test_noisy_df,
                split_name=f"{lookup_type}_{input_variant}_test_noisy",
                input_variant=input_variant,
            )

    return all_metrics


def run_ml_experiments(
    train_aug_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_clean_df: pd.DataFrame,
    test_noisy_df: pd.DataFrame,
) -> list[pd.DataFrame]:
    all_metrics = []

    for model_type in ML_MODEL_TYPES:
        for input_variant, input_col in INPUT_VARIANTS.items():
            print(f"\nTraining model={model_type}, input_variant={input_variant}")

            models = train_models(
                train_df=train_aug_df,
                model_type=model_type,
                input_col=input_col,
            )
            save_models(models, model_type=model_type, input_variant=input_variant)

            val_metrics = evaluate_models(
                models=models,
                df=val_df,
                split_name="val_clean",
                input_col=input_col,
                model_type=model_type,
                input_variant=input_variant,
            )
            test_clean_metrics = evaluate_models(
                models=models,
                df=test_clean_df,
                split_name="test_clean",
                input_col=input_col,
                model_type=model_type,
                input_variant=input_variant,
            )
            test_noisy_metrics = evaluate_models(
                models=models,
                df=test_noisy_df,
                split_name="test_noisy",
                input_col=input_col,
                model_type=model_type,
                input_variant=input_variant,
            )

            metrics_df = pd.concat(
                [val_metrics, test_clean_metrics, test_noisy_metrics],
                ignore_index=True,
            )
            all_metrics.append(metrics_df)

            save_csv(
                metrics_df,
                METRICS_DIR / f"{model_type}_{input_variant}_metrics.csv",
            )

            pred_clean = make_prediction_table(models, test_clean_df, input_col=input_col)
            pred_noisy = make_prediction_table(models, test_noisy_df, input_col=input_col)

            save_csv(
                pred_clean,
                PREDICTIONS_DIR / f"{model_type}_{input_variant}_test_clean_predictions.csv",
            )
            save_csv(
                pred_noisy,
                PREDICTIONS_DIR / f"{model_type}_{input_variant}_test_noisy_predictions.csv",
            )

            print_classification_reports(
                models,
                test_clean_df,
                split_name=f"{model_type}_{input_variant}_test_clean",
                input_col=input_col,
            )
            print_classification_reports(
                models,
                test_noisy_df,
                split_name=f"{model_type}_{input_variant}_test_noisy",
                input_col=input_col,
            )

    return all_metrics


def main() -> None:
    # 1. Load CLSI table
    df = load_and_clean_clsi(CLSI_PATH)
    print(f"Loaded CLSI rows: {len(df)}")

    # 2. Clean data
    # already handled in load_and_clean_clsi

    # 3. Split train/val/test
    train_df, val_df, test_clean_df = split_train_val_test(df)
    print(f"Train rows: {len(train_df)}")
    print(f"Val rows: {len(val_df)}")
    print(f"Clean test rows: {len(test_clean_df)}")

    ensure_dir(PROCESSED_DIR)
    ensure_dir(METRICS_DIR)
    ensure_dir(PREDICTIONS_DIR)

    save_csv(train_df, PROCESSED_DIR / "train_clean.csv")
    save_csv(val_df, PROCESSED_DIR / "val_clean.csv")
    save_csv(test_clean_df, PROCESSED_DIR / "test_clean.csv")

    # 4. Augment training set with noisy variants
    train_aug_df = augment_training_data(train_df)
    test_noisy_df = create_noisy_test_set(test_clean_df)

    save_csv(train_aug_df, PROCESSED_DIR / "train_augmented.csv")
    save_csv(test_noisy_df, PROCESSED_DIR / "test_noisy.csv")

    all_metrics = []

    # Lookup baselines should train on clean train set
    all_metrics.extend(
        run_lookup_experiments(
            full_clsi_df=df,
            val_df=val_df,
            test_clean_df=test_clean_df,
            test_noisy_df=test_noisy_df,
        )
    )

    # ML models should train on augmented train set
    all_metrics.extend(
        run_ml_experiments(
            train_aug_df=train_aug_df,
            val_df=val_df,
            test_clean_df=test_clean_df,
            test_noisy_df=test_noisy_df,
        )
    )

    all_metrics_df = pd.concat(all_metrics, ignore_index=True)
    save_csv(all_metrics_df, METRICS_DIR / "all_metrics_summary.csv")

    summary_accuracy = all_metrics_df.pivot_table(
        index=["model_type", "input_variant", "split"],
        columns="target",
        values="accuracy",
    ).reset_index()
    save_csv(summary_accuracy, METRICS_DIR / "accuracy_comparison_table.csv")

    summary_macro_f1 = all_metrics_df.pivot_table(
        index=["model_type", "input_variant", "split"],
        columns="target",
        values="macro_f1",
    ).reset_index()
    save_csv(summary_macro_f1, METRICS_DIR / "macro_f1_comparison_table.csv")

    print("\nDone.")
    print(f"Processed data saved to: {PROCESSED_DIR}")
    print(f"Metrics saved to: {METRICS_DIR}")
    print(f"Predictions saved to: {PREDICTIONS_DIR}")


if __name__ == "__main__":
    main()