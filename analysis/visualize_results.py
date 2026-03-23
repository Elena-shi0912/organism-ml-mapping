import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
METRICS_DIR = Path("outputs/metrics")
SAVE_DIR = Path("outputs/figures")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_LABELS = {
    "closed_world_exact_lookup": "CW Exact Lookup",
    "closed_world_normalized_lookup": "CW Norm Lookup",
    "closed_world_retrieval_top1": "CW Retrieval Top-1",
    "closed_world_retrieval_top3": "CW Retrieval Top-3",
    "closed_world_lsa_retrieval_top1": "CW LSA Top-1",
    "closed_world_lsa_retrieval_top3": "CW LSA Top-3",
    "generalization_exact_lookup": "Gen Exact Lookup",
    "generalization_normalized_lookup": "Gen Norm Lookup",
    "generalization_retrieval_top1": "Gen Retrieval Top-1",
    "generalization_retrieval_top3": "Gen Retrieval Top-3",
    "generalization_lsa_retrieval_top1": "Gen LSA Top-1",
    "generalization_lsa_retrieval_top3": "Gen LSA Top-3",
    "logreg": "Logistic Regression",
    "random_forest": "Random Forest",
}

def load_all_metrics() -> pd.DataFrame:
    metric_files = [
        p for p in METRICS_DIR.glob("*.csv")
        if "comparison_table" not in p.name
        and p.name not in {"all_metrics_summary.csv", "all_metrics_with_retrieval.csv"}
    ]
    frames = [pd.read_csv(p) for p in metric_files]
    return pd.concat(frames, ignore_index=True)


def add_plot_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["model_label"] = out["model_type"].map(MODEL_LABELS).fillna(out["model_type"])
    return out


def apply_axis_format(ax, xlabel: str | None = None, ylabel: str | None = None) -> None:
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")


# Load results
df = load_all_metrics()
df = add_plot_labels(df)
df.to_csv(METRICS_DIR / "all_metrics_with_retrieval.csv", index=False)

print("Loaded metrics:", df.shape)
print(df.head())

sns.set(style="whitegrid")

############################################
# 1. Overall model comparison (accuracy)
############################################

plt.figure(figsize=(14, 7))

sns.barplot(
    data=df[df["split"]=="test_noisy"],
    x="model_label",
    y="accuracy",
    hue="input_variant"
)

ax = plt.gca()
ax.set_title("Model Accuracy on Noisy Test Data")
apply_axis_format(ax, xlabel="Model Type", ylabel="Accuracy")

plt.tight_layout()
plt.savefig(SAVE_DIR / "noisy_accuracy_comparison.png")
plt.close()

############################################
# 2. Clean vs Noisy robustness
############################################

plt.figure(figsize=(14, 7))

sns.barplot(
    data=df,
    x="model_label",
    y="accuracy",
    hue="split"
)

ax = plt.gca()
ax.set_title("Clean vs Noisy Performance")
apply_axis_format(ax, xlabel="Model Type", ylabel="Accuracy")

plt.tight_layout()
plt.savefig(SAVE_DIR / "clean_vs_noisy.png")
plt.close()

############################################
# 3. Input ablation comparison
############################################

plt.figure(figsize=(14, 7))

sns.barplot(
    data=df[df["split"]=="test_noisy"],
    x="input_variant",
    y="accuracy",
    hue="model_type"
)

ax = plt.gca()
ax.set_title("Effect of Input Features")
apply_axis_format(ax, xlabel="Input Variant", ylabel="Accuracy")

plt.tight_layout()
plt.savefig(SAVE_DIR / "input_ablation.png")
plt.close()

############################################
# 4. Per-target performance
############################################

g = sns.catplot(
    data=df[df["split"]=="test_noisy"],
    x="model_label",
    y="accuracy",
    hue="input_variant",
    col="target",
    kind="bar",
    height=5,
    aspect=1.2,
    sharex=False
)

g.fig.suptitle("Performance by Prediction Target", y=1.05)
for ax in g.axes.flat:
    apply_axis_format(ax, xlabel="Model Type", ylabel="Accuracy")
g.fig.tight_layout()

g.fig.savefig(SAVE_DIR / "target_breakdown.png")
plt.close()

############################################
# 5. Robustness drop (clean → noisy)
############################################

pivot = df.pivot_table(
    index=["model_type","input_variant","target"],
    columns="split",
    values="accuracy"
).reset_index()
pivot["model_label"] = pivot["model_type"].map(MODEL_LABELS).fillna(pivot["model_type"])

pivot["robustness_drop"] = pivot["test_clean"] - pivot["test_noisy"]

plt.figure(figsize=(14, 7))

sns.barplot(
    data=pivot,
    x="model_label",
    y="robustness_drop",
    hue="input_variant"
)

ax = plt.gca()
ax.set_title("Robustness Drop (Clean → Noisy)")
apply_axis_format(ax, xlabel="Model Type", ylabel="robustness_drop")
plt.tight_layout()
plt.savefig(SAVE_DIR / "robustness_drop.png")

print("Figures saved to:", SAVE_DIR)
