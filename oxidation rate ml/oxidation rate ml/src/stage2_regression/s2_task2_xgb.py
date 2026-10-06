import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR   = _ROOT / "data"
MODELS_DIR = _ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TARGET = 'log10_k'
SEED   = 42

FEATURE_COLS = [
    'Al', 'Si', 'Ti', 'V', 'Cr', 'Zr', 'Nb', 'Mo', 'Hf', 'Ta', 'W',
    'Tm', 'delta', 'To', 't',
    'protective_sum', 'risk_sum', 'Al_Cr_interaction', 'thermal_dose',
    'predicted_n',
]

# ── Load data ────────────────────────────────────────────────
train = pd.read_csv(f"{DATA_DIR}/train_s2_scaled.csv")
test  = pd.read_csv(f"{DATA_DIR}/test_s2_scaled.csv")

X_train = train[FEATURE_COLS].values
y_train = train[TARGET].values
X_test  = test[FEATURE_COLS].values
y_test  = test[TARGET].values

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

# ── Param grid ───────────────────────────────────────────────
param_grid = {
    'n_estimators'    : [300, 500, 700],
    'max_depth'       : [3, 4, 5],
    'learning_rate'   : [0.01, 0.03, 0.05, 0.1],
    'subsample'       : [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'reg_alpha'       : [0, 0.1, 0.5, 1.0],
    'reg_lambda'      : [1, 2, 5],
    'min_child_weight': [1, 3, 5],
}

base = XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmse',
    verbosity=0,
    random_state=SEED,
    n_jobs=1,
)

search = RandomizedSearchCV(
    base, param_grid,
    n_iter=20, cv=3,
    scoring='neg_root_mean_squared_error',
    random_state=SEED, n_jobs=1, refit=True,
    verbose=1,
)
search.fit(X_train, y_train)
best = search.best_estimator_

# ── CV R² (separate pass with r2 scoring) ────────────────────
cv_r2_scores = cross_val_score(best, X_train, y_train, cv=3, scoring='r2')
cv_r2        = cv_r2_scores.mean()

# ── Test metrics on log10_k ───────────────────────────────────
y_pred_log = best.predict(X_test)
test_r2_log = r2_score(y_test, y_pred_log)

# ── Back-transform to raw k and compute metrics ───────────────
y_test_raw  = 10 ** y_test
y_pred_raw  = 10 ** y_pred_log

test_r2_raw  = r2_score(y_test_raw, y_pred_raw)
test_rmse_raw = np.sqrt(mean_squared_error(y_test_raw, y_pred_raw))
test_mae_raw  = mean_absolute_error(y_test_raw, y_pred_raw)

# ── Report ───────────────────────────────────────────────────
print("\n" + "=" * 55)
print("XGBoost Regressor (Stage 2) — Results")
print("=" * 55)
print(f"Best hyperparameters     : {search.best_params_}")
print(f"CV RMSE (log10_k, neg)   : {search.best_score_:.4f}")
print(f"CV R²   (log10_k)        : {cv_r2:.4f}")
print(f"Test R² (log10_k)        : {test_r2_log:.4f}")
print(f"Test R² (raw k)          : {test_r2_raw:.4f}")
print(f"Test RMSE (raw k)        : {test_rmse_raw:.4f}")
print(f"Test MAE  (raw k)        : {test_mae_raw:.4f}")

if test_r2_raw > 0.95:
    print("\nWARNING: Test R² on raw k > 0.95 — possible data leakage. Investigate.")

# ── Save model ───────────────────────────────────────────────
with open(f"{MODELS_DIR}/model_xgb_s2.pkl", 'wb') as f:
    pickle.dump(best, f)
print(f"\nModel saved: model_xgb_s2.pkl")
