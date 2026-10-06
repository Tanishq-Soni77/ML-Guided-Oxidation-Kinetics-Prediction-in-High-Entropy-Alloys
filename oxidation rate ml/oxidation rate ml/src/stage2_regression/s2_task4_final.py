import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR    = _ROOT / "data"
MODELS_DIR  = _ROOT / "models"
METRICS_DIR = _ROOT / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
TARGET = 'log10_k'

FEATURE_COLS = [
    'Al', 'Si', 'Ti', 'V', 'Cr', 'Zr', 'Nb', 'Mo', 'Hf', 'Ta', 'W',
    'Tm', 'delta', 'To', 't',
    'protective_sum', 'risk_sum', 'Al_Cr_interaction', 'thermal_dose',
    'predicted_n',
]

# ── Load scaled data ──────────────────────────────────────────
train_sc = pd.read_csv(f"{DATA_DIR}/train_s2_scaled.csv")
test_sc  = pd.read_csv(f"{DATA_DIR}/test_s2_scaled.csv")

X_train = train_sc[FEATURE_COLS].values
y_train = train_sc[TARGET].values
X_test  = test_sc[FEATURE_COLS].values
y_test  = test_sc[TARGET].values

# ── Load original data (for metadata columns) ─────────────────
test_orig = pd.read_csv(f"{DATA_DIR}/test_data_s2.csv")

# ── Load models ───────────────────────────────────────────────
model_files = {
    'XGBoost'     : 'model_xgb_s2.pkl',
    'RandomForest': 'model_rf_s2.pkl',
    'ExtraTrees'  : 'model_et_s2.pkl',
}
models = {}
for name, fname in model_files.items():
    with open(f"{MODELS_DIR}/{fname}", 'rb') as f:
        models[name] = pickle.load(f)
    print(f"Loaded: {fname}")

# ─────────────────────────────────────────────────────────────
# Step 1 — Compare all 3 models on test set
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Step 1 — Model comparison (test R² on log10_k)")
print("=" * 65)

def compute_metrics(model, X, y_log):
    y_pred_log = model.predict(X)
    y_raw      = 10 ** y_log
    y_pred_raw = 10 ** y_pred_log
    return {
        'r2_log'  : r2_score(y_log, y_pred_log),
        'r2_raw'  : r2_score(y_raw, y_pred_raw),
        'rmse_raw': np.sqrt(mean_squared_error(y_raw, y_pred_raw)),
        'mae_raw' : mean_absolute_error(y_raw, y_pred_raw),
        'y_pred_log': y_pred_log,
    }

results = {}
for name, model in models.items():
    results[name] = compute_metrics(model, X_test, y_test)

print(f"\n{'Model':<14}  {'R²(log10_k)':>11}  {'R²(raw k)':>9}  "
      f"{'RMSE(raw k)':>11}  {'MAE(raw k)':>10}")
print("-" * 62)
for name, m in results.items():
    print(f"{name:<14}  {m['r2_log']:>11.4f}  {m['r2_raw']:>9.4f}  "
          f"{m['rmse_raw']:>11.4f}  {m['mae_raw']:>10.4f}")

winner_name  = max(results, key=lambda n: results[n]['r2_log'])
winner_model = models[winner_name]
winner_m     = results[winner_name]

print(f"\nWINNER: {winner_name}  (Test R² log10_k = {winner_m['r2_log']:.4f})")

# ─────────────────────────────────────────────────────────────
# Step 2 — Final predictions using winner
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"Step 2 — Final predictions using {winner_name}")
print("=" * 65)

pred_log10k = winner_m['y_pred_log']
actual_log10k = y_test
predicted_k   = 10 ** pred_log10k
actual_k      = 10 ** actual_log10k

print(f"Predicted log10_k range: [{pred_log10k.min():.4f}, {pred_log10k.max():.4f}]")
print(f"Actual    log10_k range: [{actual_log10k.min():.4f}, {actual_log10k.max():.4f}]")

# ─────────────────────────────────────────────────────────────
# Step 3 — Save final_predictions.csv
# ─────────────────────────────────────────────────────────────
final_df = pd.DataFrame({
    'alloy_group'      : test_sc['alloy_group'].values,
    'kinetics_class'   : test_sc['kinetics_class'].values,
    'actual_k'         : actual_k,
    'predicted_k'      : predicted_k,
    'actual_log10k'    : actual_log10k,
    'predicted_log10k' : pred_log10k,
})
final_df.to_csv(f"{METRICS_DIR}/final_predictions.csv", index=False)
print(f"\nSaved: final_predictions.csv  ({len(final_df)} rows)")

# ─────────────────────────────────────────────────────────────
# Step 4 — Final summary
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("FINAL SUMMARY — Stage 2 Regression")
print("=" * 65)
print(f"Winning model        : {winner_name}")
print(f"Hyperparameters      : {winner_model.get_params()}")
print(f"Test R²  (log10_k)   : {winner_m['r2_log']:.4f}")
print(f"Test R²  (raw k)     : {winner_m['r2_raw']:.4f}")
print(f"Test RMSE (raw k)    : {winner_m['rmse_raw']:.4f}")
print(f"Test MAE  (raw k)    : {winner_m['mae_raw']:.4f}")
