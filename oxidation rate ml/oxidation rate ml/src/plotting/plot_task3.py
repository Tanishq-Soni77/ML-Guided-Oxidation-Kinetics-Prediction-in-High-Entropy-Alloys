import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import make_scorer, f1_score
from xgboost import XGBRegressor

from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR  = _ROOT / "data"
PLOTS_DIR = _ROOT / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42

S2_FEATURE_COLS = [
    'Al', 'Si', 'Ti', 'V', 'Cr', 'Zr', 'Nb', 'Mo', 'Hf', 'Ta', 'W',
    'Tm', 'delta', 'To', 't',
    'protective_sum', 'risk_sum', 'Al_Cr_interaction', 'thermal_dose',
    'predicted_n',
]

# ── Load data ─────────────────────────────────────────────────
train_s1 = pd.read_csv(f"{DATA_DIR}/train_scaled_3class.csv")
test_s1  = pd.read_csv(f"{DATA_DIR}/test_scaled_3class.csv")
train_s2 = pd.read_csv(f"{DATA_DIR}/train_s2_scaled.csv")
test_s2  = pd.read_csv(f"{DATA_DIR}/test_s2_scaled.csv")

# Stage 1 arrays
s1_feat_cols = [c for c in train_s1.columns if c != 'kinetics_class']
X_train_s1   = train_s1[s1_feat_cols].values
y_train_s1   = train_s1['kinetics_class'].values
X_test_s1    = test_s1[s1_feat_cols].values
y_test_s1    = test_s1['kinetics_class'].values

# Stage 2 arrays
X_train_s2_raw = train_s2[S2_FEATURE_COLS].values
y_train_s2     = train_s2['log10_k'].values
X_test_s2_raw  = test_s2[S2_FEATURE_COLS].values
y_test_s2      = test_s2['log10_k'].values

print(f"S1 — Train: {X_train_s1.shape}  Test: {X_test_s1.shape}")
print(f"S2 — Train: {X_train_s2_raw.shape}  Test: {X_test_s2_raw.shape}")

# ─────────────────────────────────────────────────────────────
# Stage 1 — Retrain ExtraTrees + permutation importance
# ─────────────────────────────────────────────────────────────
print("\nTraining Stage 1 ExtraTrees…")
et = ExtraTreesClassifier(
    n_estimators      = 600,
    min_samples_split = 5,
    max_depth         = None,
    class_weight      = 'balanced',
    random_state      = SEED,
    n_jobs            = -1,
)
et.fit(X_train_s1, y_train_s1)

f1_macro_scorer = make_scorer(f1_score, average='macro', zero_division=0)
pi_s1 = permutation_importance(
    et, X_test_s1, y_test_s1,
    scoring    = f1_macro_scorer,
    n_repeats  = 10,
    random_state = SEED,
    n_jobs     = -1,
)
s1_imp = pd.DataFrame({
    'feature'   : s1_feat_cols,
    'importance': pi_s1.importances_mean,
    'std'       : pi_s1.importances_std,
}).sort_values('importance', ascending=False).head(10)
print("Stage 1 permutation importance done.")

# ─────────────────────────────────────────────────────────────
# Stage 2 — RobustScaler + retrain XGBoost + permutation importance
# ─────────────────────────────────────────────────────────────
print("Training Stage 2 XGBoost…")
scaler         = RobustScaler()
X_train_s2_sc  = scaler.fit_transform(X_train_s2_raw)
X_test_s2_sc   = scaler.transform(X_test_s2_raw)

xgb = XGBRegressor(
    n_estimators    = 500,
    max_depth       = 3,
    learning_rate   = 0.1,
    subsample       = 0.8,
    colsample_bytree= 0.8,
    reg_alpha       = 0,
    reg_lambda      = 1,
    min_child_weight= 3,
    objective       = 'reg:squarederror',
    eval_metric     = 'rmse',
    verbosity       = 0,
    random_state    = SEED,
    n_jobs          = 1,
)
xgb.fit(X_train_s2_sc, y_train_s2)

pi_s2 = permutation_importance(
    xgb, X_test_s2_sc, y_test_s2,
    scoring      = 'r2',
    n_repeats    = 10,
    random_state = SEED,
    n_jobs       = -1,
)
s2_imp = pd.DataFrame({
    'feature'   : S2_FEATURE_COLS,
    'importance': pi_s2.importances_mean,
    'std'       : pi_s2.importances_std,
}).sort_values('importance', ascending=False).head(10)
print("Stage 2 permutation importance done.")

# ─────────────────────────────────────────────────────────────
# Plot 5 — Side-by-side horizontal bar charts
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

def plot_importance(ax, imp_df, title, xlabel, color):
    # Plot in descending order (most important at top)
    imp_sorted = imp_df.sort_values('importance', ascending=True)
    ax.barh(
        imp_sorted['feature'],
        imp_sorted['importance'],
        xerr=imp_sorted['std'],
        color=color,
        alpha=0.82,
        edgecolor='white',
        capsize=3,
        error_kw=dict(ecolor='dimgrey', lw=0.8, capthick=0.8),
    )
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', labelsize=9)

plot_importance(
    axes[0], s1_imp,
    title  = "Stage 1 Permutation Importance",
    xlabel = "Mean decrease in macro-F1",
    color  = 'steelblue',
)
plot_importance(
    axes[1], s2_imp,
    title  = "Stage 2 Permutation Importance",
    xlabel = r"Mean decrease in $R^2$",
    color  = 'darkorange',
)

plt.suptitle("Permutation Feature Importance — Stage 1 (ExtraTrees) & Stage 2 (XGBoost)",
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/plot5_permutation_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plot5_permutation_importance.png")
