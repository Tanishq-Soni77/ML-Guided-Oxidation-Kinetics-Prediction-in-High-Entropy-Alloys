import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR  = _ROOT / "data"
PLOTS_DIR = _ROOT / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TARGET = 'log10_k'
SEED   = 42

FEATURE_COLS = [
    'Al', 'Si', 'Ti', 'V', 'Cr', 'Zr', 'Nb', 'Mo', 'Hf', 'Ta', 'W',
    'Tm', 'delta', 'To', 't',
    'protective_sum', 'risk_sum', 'Al_Cr_interaction', 'thermal_dose',
    'predicted_n',
]

# ── Load data ─────────────────────────────────────────────────
train = pd.read_csv(f"{DATA_DIR}/train_data_s2.csv")
test  = pd.read_csv(f"{DATA_DIR}/test_data_s2.csv")

train = train.rename(columns={'predicted_n_train': 'predicted_n'})
test  = test.rename(columns={'predicted_n_test':  'predicted_n'})

X_train_raw = train[FEATURE_COLS].values
y_train     = train[TARGET].values
X_test_raw  = test[FEATURE_COLS].values

print(f"Train: {X_train_raw.shape}  |  Test: {X_test_raw.shape}")

# ── RobustScaler — keep as DataFrames with feature names ──────
scaler = RobustScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_raw), columns=FEATURE_COLS
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_raw), columns=FEATURE_COLS
)

# ── Retrain XGBoost ───────────────────────────────────────────
model = XGBRegressor(
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
model.fit(X_train_scaled, y_train)
print("Model trained on full train set.")

# ── SHAP: extract booster and fix base_score ──────────────────
booster   = model.get_booster()
explainer = shap.TreeExplainer(booster)
shap_values = explainer.shap_values(X_test_scaled)
print(f"SHAP values computed. Shape: {np.array(shap_values).shape}")

# ── Beeswarm via summary_plot (top 15 features) ───────────────
shap.summary_plot(
    shap_values,
    X_test_scaled,
    max_display=15,
    show=False,
)
plt.savefig(f"{PLOTS_DIR}/shap_beeswarm.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: shap_beeswarm.png")

# ── Top 5 features by mean absolute SHAP value ───────────────
mean_abs_shap = (
    pd.DataFrame(np.abs(shap_values), columns=FEATURE_COLS)
    .mean()
    .sort_values(ascending=False)
)

print("\nTop 5 features by mean absolute SHAP value:")
print(f"{'Rank':<5}  {'Feature':<22}  {'Mean |SHAP|':>11}")
print("-" * 42)
for rank, (feature, value) in enumerate(mean_abs_shap.head(5).items(), start=1):
    print(f"{rank:<5}  {feature:<22}  {value:>11.4f}")
