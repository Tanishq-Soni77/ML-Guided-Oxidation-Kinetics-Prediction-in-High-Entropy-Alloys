import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix

from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR  = _ROOT / "data"
PLOTS_DIR = _ROOT / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TARGET      = 'kinetics_class'
CLASS_NAMES = ['Linear', 'Parabolic', 'HigherOrder']
SEED        = 42

# ── Load data ─────────────────────────────────────────────────
train_3c = pd.read_csv(f"{DATA_DIR}/train_scaled_3class.csv")
test_3c  = pd.read_csv(f"{DATA_DIR}/test_scaled_3class.csv")

feature_cols = [c for c in train_3c.columns if c != TARGET]

X_train = train_3c[feature_cols].values
y_train = train_3c[TARGET].values
X_test  = test_3c[feature_cols].values
y_test  = test_3c[TARGET].values

# ─────────────────────────────────────────────────────────────
# Plot 1 — Class distribution before vs after merging
# ─────────────────────────────────────────────────────────────
before_names  = ['Linear', 'Parabolic', 'Cubic', 'Quartic']
before_counts = [108, 111, 9, 6]

after_names   = ['Linear', 'Parabolic', 'HigherOrder']
after_counts  = [108, 111, 15]

fig, ax = plt.subplots(figsize=(9, 5))

x_before = np.arange(len(before_names))
x_after  = np.arange(len(after_names))

# Plot before (left group) and after (right group) side by side
bar_width = 0.35
group_gap = 0.5   # horizontal gap between the two groups

x_b = np.arange(len(before_names))
x_a = np.arange(len(after_names)) + len(before_names) + group_gap

bars_before = ax.bar(x_b, before_counts, width=bar_width * 1.5,
                     color='steelblue', alpha=0.85, label='Before merging (4-class)')
bars_after  = ax.bar(x_a, after_counts,  width=bar_width * 1.5,
                     color='darkorange', alpha=0.85, label='After merging (3-class)')

# Annotate bar heights
for bar in bars_before:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            str(int(bar.get_height())), ha='center', va='bottom', fontsize=10)
for bar in bars_after:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            str(int(bar.get_height())), ha='center', va='bottom', fontsize=10)

all_ticks  = list(x_b) + list(x_a)
all_labels = before_names + after_names
ax.set_xticks(all_ticks)
ax.set_xticklabels(all_labels, fontsize=11)

ax.set_ylabel("Count", fontsize=12)
ax.set_title("Class Distribution: Before vs After Class Merging (Train Set)",
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim(0, max(before_counts + after_counts) * 1.15)
ax.axvline(x=len(before_names) - 1 + group_gap / 2, color='grey',
           linestyle='--', linewidth=0.8, alpha=0.6)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/plot1_class_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plot1_class_distribution.png")

# ─────────────────────────────────────────────────────────────
# Plot 2 — Normalized confusion matrix (ExtraTrees 3-class)
# ─────────────────────────────────────────────────────────────
model = ExtraTreesClassifier(
    n_estimators      = 600,
    min_samples_split = 5,
    max_depth         = None,
    class_weight      = 'balanced',
    random_state      = SEED,
    n_jobs            = -1,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cm      = confusion_matrix(y_test, y_pred)
cm_norm = (cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100).astype(int)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    linewidths=0.5,
    linecolor='grey',
    ax=ax,
    cbar_kws={'label': '% of true class'},
)
ax.set_xlabel("Predicted label", fontsize=12)
ax.set_ylabel("True label", fontsize=12)
ax.set_title("Stage 1 Confusion Matrix — ExtraTrees (3-class, normalized)",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/plot2_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plot2_confusion_matrix.png")
