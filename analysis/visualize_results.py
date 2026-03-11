import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
METRICS_PATH = Path("outputs/metrics/all_metrics_summary.csv")
SAVE_DIR = Path("outputs/figures")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Load results
df = pd.read_csv(METRICS_PATH)

print("Loaded metrics:", df.shape)
print(df.head())

sns.set(style="whitegrid")

############################################
# 1. Overall model comparison (accuracy)
############################################

plt.figure(figsize=(10,6))

sns.barplot(
    data=df[df["split"]=="test_noisy"],
    x="model_type",
    y="accuracy",
    hue="input_variant"
)

plt.title("Model Accuracy on Noisy Test Data")
plt.ylabel("Accuracy")
plt.xlabel("Model Type")

plt.tight_layout()
plt.savefig(SAVE_DIR / "noisy_accuracy_comparison.png")
plt.close()

############################################
# 2. Clean vs Noisy robustness
############################################

plt.figure(figsize=(10,6))

sns.barplot(
    data=df,
    x="model_type",
    y="accuracy",
    hue="split"
)

plt.title("Clean vs Noisy Performance")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.savefig(SAVE_DIR / "clean_vs_noisy.png")
plt.close()

############################################
# 3. Input ablation comparison
############################################

plt.figure(figsize=(10,6))

sns.barplot(
    data=df[df["split"]=="test_noisy"],
    x="input_variant",
    y="accuracy",
    hue="model_type"
)

plt.title("Effect of Input Features")
plt.ylabel("Accuracy")
plt.xlabel("Input Variant")

plt.tight_layout()
plt.savefig(SAVE_DIR / "input_ablation.png")
plt.close()

############################################
# 4. Per-target performance
############################################

g = sns.catplot(
    data=df[df["split"]=="test_noisy"],
    x="model_type",
    y="accuracy",
    hue="input_variant",
    col="target",
    kind="bar",
    height=5,
    aspect=1
)

g.fig.suptitle("Performance by Prediction Target", y=1.05)

plt.savefig(SAVE_DIR / "target_breakdown.png")
plt.close()

############################################
# 5. Robustness drop (clean → noisy)
############################################

pivot = df.pivot_table(
    index=["model_type","input_variant","target"],
    columns="split",
    values="accuracy"
).reset_index()

pivot["robustness_drop"] = pivot["test_clean"] - pivot["test_noisy"]

sns.barplot(
    data=pivot,
    x="model_type",
    y="robustness_drop",
    hue="input_variant"
)

plt.title("Robustness Drop (Clean → Noisy)")
plt.savefig(SAVE_DIR / "robustness_drop.png")

print("Figures saved to:", SAVE_DIR)