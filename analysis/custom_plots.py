import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

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
    out["setting"] = "other"
    out.loc[out["model_type"].str.startswith("closed_world"), "setting"] = "closed_world"
    out.loc[out["model_type"].str.startswith("generalization"), "setting"] = "generalization"
    out["model_family"] = out["model_type"].str.replace("^closed_world_", "", regex=True)
    out["model_family"] = out["model_family"].str.replace("^generalization_", "", regex=True)
    return out


def apply_axis_format(ax, xlabel: str | None = None, ylabel: str | None = None) -> None:
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

def plot_grouped_bars(
    ax,
    df: pd.DataFrame,
    x_col: str,
    series_col: str,
    y_col: str,
    order: list[str] | None = None,
    series_order: list[str] | None = None,
    title: str | None = None,
):
    pivot = df.pivot_table(index=x_col, columns=series_col, values=y_col, aggfunc="mean")
    if order is not None:
        pivot = pivot.reindex(order)
    if series_order is not None:
        pivot = pivot.reindex(columns=series_order)

    x_labels = pivot.index.tolist()
    series_labels = pivot.columns.tolist()
    values = pivot.values

    n_series = len(series_labels)
    x = np.arange(len(x_labels))
    width = 0.8 / max(n_series, 1)

    bar_containers = []
    for i, label in enumerate(series_labels):
        bars = ax.bar(x + i * width, values[:, i], width=width, label=label)
        bar_containers.append(bars)

    ax.set_xticks(x + (n_series - 1) * width / 2)
    ax.set_xticklabels(x_labels)
    if title:
        ax.set_title(title)
    ax.legend(title=series_col, fontsize=8)
    return {
        "x_labels": x_labels,
        "series_labels": series_labels,
        "values": values,
        "bar_containers": bar_containers,
        "x_positions": x,
        "width": width,
    }


def annotate_lookup_collapse(ax, bar_info, x_label: str, series_label: str, text: str):
    x_labels = bar_info["x_labels"]
    series_labels = bar_info["series_labels"]
    values = bar_info["values"]
    x_positions = bar_info["x_positions"]
    width = bar_info["width"]

    if x_label not in x_labels or series_label not in series_labels:
        return

    xi = x_labels.index(x_label)
    si = series_labels.index(series_label)
    x = x_positions[xi] + si * width
    y = values[xi, si]
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(x + 0.6, min(1.05, y + 0.35)),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "lw": 1.2},
        fontsize=9,
        ha="left",
    )


def annotate_best_family(ax, bar_info, series_label: str, text: str):
    series_labels = bar_info["series_labels"]
    if series_label not in series_labels:
        return

    si = series_labels.index(series_label)
    values = bar_info["values"][:, si]
    x_positions = bar_info["x_positions"]
    width = bar_info["width"]
    xi = int(np.argmax(values))
    x = x_positions[xi] + si * width
    y = values[xi]
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(x + 0.4, min(1.05, y + 0.2)),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "lw": 1.2},
        fontsize=9,
        ha="left",
    )


df = add_plot_labels(load_all_metrics())
closed_world_or_ml = df[
    (df["model_type"].str.startswith("closed_world"))
    | (df["model_type"].isin(["logreg", "random_forest"]))
]

# 1. Effect of noisy inputs on each target for each model (closed-world only, input_variant=code_only)
noisy_effect = closed_world_or_ml[
    (closed_world_or_ml["input_variant"] == "code_only")
    & (closed_world_or_ml["split"].isin(["test_clean", "test_noisy"]))
]

targets = noisy_effect["target"].unique().tolist()
fig, axes = plt.subplots(1, len(targets), figsize=(18, 5), sharey=True)
if len(targets) == 1:
    axes = [axes]
for ax, target in zip(axes, targets):
    subset = noisy_effect[noisy_effect["target"] == target]
    bar_info = plot_grouped_bars(
        ax,
        subset,
        x_col="model_label",
        series_col="split",
        y_col="accuracy",
        series_order=["test_clean", "test_noisy"],
        title=target,
    )
    annotate_lookup_collapse(
        ax,
        bar_info,
        x_label="CW Exact Lookup",
        series_label="test_noisy",
        text="Lookup collapses\nunder noise",
    )
    annotate_best_family(
        ax,
        bar_info,
        series_label="test_noisy",
        text="Retrieval/LSA\nremain high",
    )
    apply_axis_format(ax, xlabel="Model Type", ylabel="Accuracy")
fig.suptitle("Closed-World: Clean vs Noisy (code_only)")
fig.tight_layout()
fig.savefig(SAVE_DIR / "noisy_effect_closed_world_code.png")
plt.close(fig)

# 2. Robustness to noisy inputs on each target (closed-world only, input_variant=code_only)
pivot = closed_world_or_ml[
    (closed_world_or_ml["input_variant"] == "code_only")
].pivot_table(
    index=["model_type", "model_label", "target"],
    columns="split",
    values="accuracy",
).reset_index()

pivot["robustness_drop"] = pivot["test_clean"] - pivot["test_noisy"]

targets = pivot["target"].unique().tolist()
fig, axes = plt.subplots(1, len(targets), figsize=(18, 5), sharey=True)
if len(targets) == 1:
    axes = [axes]
for ax, target in zip(axes, targets):
    subset = pivot[pivot["target"] == target]
    x_labels = subset["model_label"].tolist()
    y_vals = subset["robustness_drop"].tolist()
    x = np.arange(len(x_labels))
    ax.bar(x, y_vals, width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_title(target)
    if "CW Exact Lookup" in x_labels:
        i = x_labels.index("CW Exact Lookup")
        ax.annotate(
            "Largest drop",
            xy=(x[i], y_vals[i]),
            xytext=(x[i] + 0.6, min(1.05, y_vals[i] + 0.25)),
            textcoords="data",
            arrowprops={"arrowstyle": "->", "lw": 1.2},
            fontsize=9,
            ha="left",
        )
    if "CW Retrieval Top-1" in x_labels:
        j = x_labels.index("CW Retrieval Top-1")
        ax.annotate(
            "Most robust",
            xy=(x[j], y_vals[j]),
            xytext=(x[j] + 0.6, min(1.05, y_vals[j] + 0.2)),
            textcoords="data",
            arrowprops={"arrowstyle": "->", "lw": 1.2},
            fontsize=9,
            ha="left",
        )
    apply_axis_format(ax, xlabel="Model Type", ylabel="Robustness Drop")
fig.suptitle("Closed-World Robustness Drop (Clean → Noisy, code_only)")
fig.tight_layout()
fig.savefig(SAVE_DIR / "robustness_drop_closed_world_code.png")
plt.close(fig)

# 3. Effect of generalization on each target for each model (input_variant=code_only)
# Compare closed-world vs generalization on noisy test

gen_effect = df[
    (df["setting"].isin(["closed_world", "generalization"]))
    & (df["input_variant"] == "code_only")
    & (df["split"] == "test_noisy")
]

targets = gen_effect["target"].unique().tolist()
fig, axes = plt.subplots(1, len(targets), figsize=(18, 5), sharey=True)
if len(targets) == 1:
    axes = [axes]
for ax, target in zip(axes, targets):
    subset = gen_effect[gen_effect["target"] == target]
    plot_grouped_bars(
        ax,
        subset,
        x_col="model_family",
        series_col="setting",
        y_col="accuracy",
        series_order=["closed_world", "generalization"],
        title=target,
    )
    apply_axis_format(ax, xlabel="Model Family", ylabel="Accuracy")
fig.suptitle("Generalization Effect (Closed-World vs Generalization, code_only, noisy test)")
fig.tight_layout()
fig.savefig(SAVE_DIR / "generalization_effect_code.png")
plt.close(fig)

# 4. Effect of input variants on models (target=Breakpoint Group, generalization only)
input_effect = df[
    (df["setting"] == "generalization")
    & (df["target"] == "CLSI Breakpoint Group")
    & (df["split"] == "test_noisy")
]

fig, ax = plt.subplots(figsize=(14, 6))
plot_grouped_bars(
    ax,
    input_effect,
    x_col="model_label",
    series_col="input_variant",
    y_col="accuracy",
    series_order=["code_only", "name_only", "code_plus_name"],
    title="Generalization: Input Variant Effect (CLSI Breakpoint Group, noisy test)",
)
apply_axis_format(ax, xlabel="Model Type", ylabel="Accuracy")
fig.tight_layout()
fig.savefig(SAVE_DIR / "input_variant_effect_gen_breakpoint.png")
plt.close(fig)

print("Saved figures to:", SAVE_DIR)
