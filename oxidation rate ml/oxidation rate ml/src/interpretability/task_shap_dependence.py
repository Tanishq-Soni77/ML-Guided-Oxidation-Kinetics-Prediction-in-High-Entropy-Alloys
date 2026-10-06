import warnings
warnings.filterwarnings('ignore')

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

DEPENDENCE_FEATURES = {
    'To'           : 'shap_dep_To.png',
    'Al'           : 'shap_dep_Al.png',
    'predicted_n'  : 'shap_dep_predicted_n.png',
    'protective_sum': 'shap_dep_protective_sum.png',
}

# ── Load data ─────────────────────────────────────────────────
train = pd.read_csv(f"{DATA_DIR}/train_data_s2.csv")
test  = pd.read_csv(f"{DATA_DIR}/test_data_s2.csv")

train = train.rename(columns={'predicted_n_train': 'predicted_n'})
test  = test.rename(columns={'predicted_n_test':  'predicted_n'})

X_train_raw = train[FEATURE_COLS].values
y_train     = train[TARGET].values
X_test_raw  = test[FEATURE_COLS].values

print(f"Train: {X_train_raw.shape}  |  Test: {X_test_raw.shape}")

# ─────────────────────────────────────────────────────────────
# Step 1 — RobustScaler + retrain XGBoost on full train
# ─────────────────────────────────────────────────────────────
scaler     = RobustScaler()
X_train_sc = scaler.fit_transform(X_train_raw)
X_test_sc  = scaler.transform(X_test_raw)

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
model.fit(X_train_sc, y_train)
print("Model trained on full train set.")

# ─────────────────────────────────────────────────────────────
# Step 2 — SHAP TreeExplainer on test set
# ─────────────────────────────────────────────────────────────
X_test_df   = pd.DataFrame(X_test_sc, columns=FEATURE_COLS)
explainer   = shap.TreeExplainer(model)
shap_values = explainer(X_test_df)
print(f"SHAP values computed. Shape: {shap_values.values.shape}")

# ─────────────────────────────────────────────────────────────
# Step 3 — Dependence plots for 4 features
# ─────────────────────────────────────────────────────────────
for feature_name, fname in DEPENDENCE_FEATURES.items():
    shap.plots.scatter(shap_values[:, feature_name], show=False)
    plt.savefig(f"{PLOTS_DIR}/{fname}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")

print("\nAll 4 dependence plots saved.")
