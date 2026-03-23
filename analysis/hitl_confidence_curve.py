import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CLSI_PATH, TARGET_COLS
from src.preprocessing import load_and_clean_clsi
from src.split import split_train_val_test
from src.add_noise import create_noisy_test_set
from src.retrieval_baseline import build_retrieval_index, retrieve_topk, predict_retrieval


FIGURES_DIR = ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def get_split_data(split: str):
    df = load_and_clean_clsi(CLSI_PATH)
    train_df, val_df, test_clean_df = split_train_val_test(df)
    if split == "test_clean":
        return train_df, val_df, test_clean_df
    if split == "test_noisy":
        test_noisy_df = create_noisy_test_set(test_clean_df)
        return train_df, val_df, test_noisy_df
    raise ValueError("split must be test_clean or test_noisy")


def compute_curve(
    index_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    input_variant: str,
    num_thresholds: int,
) -> tuple[np.ndarray, np.ndarray]:
    index = build_retrieval_index(
        full_clsi_df=index_df,
        input_variant=input_variant,
        normalize=True,
    )

    candidates = retrieve_topk(index, eval_df, k=1)
    sample_target = TARGET_COLS[0]
    top1_scores = np.array([row[0][1] for row in candidates[sample_target]])

    preds = predict_retrieval(index, eval_df, k=1)
    thresholds = np.linspace(0, 1, num_thresholds)

    coverages = []
    accuracies = []
    for t in thresholds:
        mask = top1_scores >= t
        coverage = float(mask.mean())
        if coverage == 0:
            coverages.append(0.0)
            accuracies.append(np.nan)
            continue

        per_target_acc = []
        for target in TARGET_COLS:
            y_true = eval_df[target].values
            y_pred = np.array(preds[target])
            acc = (y_true[mask] == y_pred[mask]).mean()
            per_target_acc.append(acc)
        coverages.append(coverage)
        accuracies.append(float(np.mean(per_target_acc)))

    return np.array(coverages), np.array(accuracies)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HITL confidence/abstain curve.")
    parser.add_argument("--split", default="test_noisy")
    parser.add_argument("--input_variant", default="code_plus_name")
    parser.add_argument("--setting", default="closed_world", choices=["closed_world", "generalization"])
    parser.add_argument("--thresholds", type=int, default=50)
    args = parser.parse_args()

    train_df, val_df, eval_df = get_split_data(args.split)
    if args.setting == "closed_world":
        index_df = pd.concat([train_df, val_df, eval_df], ignore_index=True)
    else:
        index_df = pd.concat([train_df, val_df], ignore_index=True)

    coverages, accuracies = compute_curve(
        index_df=index_df,
        eval_df=eval_df,
        input_variant=args.input_variant,
        num_thresholds=args.thresholds,
    )

    plt.figure(figsize=(8, 6))
    plt.plot(coverages, accuracies, marker="o", linewidth=2)
    plt.title(
        f"Confidence/Abstain Curve ({args.split})\n"
        f"{args.setting} | {args.input_variant}"
    )
    plt.xlabel("Coverage (fraction auto-accepted)")
    plt.ylabel("Macro Accuracy (accepted only)")
    plt.ylim(0, 1.0)
    plt.xlim(0, 1.0)
    plt.grid(True, alpha=0.3)
    out_path = FIGURES_DIR / f"hitl_curve_{args.setting}_{args.input_variant}_{args.split}.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
