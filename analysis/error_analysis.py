import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


OUTPUTS_DIR = Path("outputs")
METRICS_PATH = OUTPUTS_DIR / "metrics" / "all_metrics_summary.csv"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
FIGURES_DIR = OUTPUTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def sanitize(text: str) -> str:
    return text.lower().replace(" ", "_")


def load_best_model(
    split: str,
    target: str,
) -> tuple[str, str]:
    metrics = pd.read_csv(METRICS_PATH)
    sub = metrics[(metrics["split"] == split) & (metrics["target"] == target)].copy()
    if sub.empty:
        raise ValueError(f"No metrics found for split={split}, target={target}")
    best = sub.sort_values("macro_f1", ascending=False).iloc[0]
    return str(best["model_type"]), str(best["input_variant"])


def compress_labels(series: pd.Series, max_labels: int) -> pd.Series:
    counts = series.value_counts()
    if len(counts) <= max_labels:
        return series
    keep = set(counts.head(max_labels).index)
    return series.where(series.isin(keep), "__OTHER__")


def build_confusion_plot(
    df: pd.DataFrame,
    target: str,
    title: str,
    max_labels: int,
    normalize: bool,
    out_path: Path,
) -> None:
    truth = compress_labels(df[target], max_labels=max_labels)
    pred_col = f"pred_{target.replace(' ', '_')}"
    preds = compress_labels(df[pred_col], max_labels=max_labels)

    labels = sorted(set(truth.unique()) | set(preds.unique()))
    cm = confusion_matrix(truth, preds, labels=labels, normalize="true" if normalize else None)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        annot=False,
        cbar=True,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate error analysis confusion matrix.")
    parser.add_argument("--split", default="test_noisy")
    parser.add_argument("--target", default="CLSI Breakpoint Group")
    parser.add_argument("--model_type", default=None)
    parser.add_argument("--input_variant", default=None)
    parser.add_argument("--max_labels", type=int, default=15)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    if args.model_type is None or args.input_variant is None:
        model_type, input_variant = load_best_model(args.split, args.target)
    else:
        model_type, input_variant = args.model_type, args.input_variant

    pred_path = (
        PREDICTIONS_DIR
        / f"{model_type}_{input_variant}_{args.split}_predictions.csv"
    )
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_path}")

    df = pd.read_csv(pred_path)
    title = (
        f"Confusion Matrix ({args.target})\n"
        f"{model_type} | {input_variant} | {args.split}"
    )
    out_path = FIGURES_DIR / f"confusion_{sanitize(args.target)}_{args.split}.png"
    build_confusion_plot(
        df=df,
        target=args.target,
        title=title,
        max_labels=args.max_labels,
        normalize=args.normalize,
        out_path=out_path,
    )
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
