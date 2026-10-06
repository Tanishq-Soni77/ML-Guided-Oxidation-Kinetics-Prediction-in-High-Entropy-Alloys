import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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

# ── Shared param grid ─────────────────────────────────────────
tree_param_grid = {
    'n_estimators'    : [300, 500],
    'max_depth'       : [None, 5, 8, 12],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf' : [1, 2, 4],
}

# ── Helper ────────────────────────────────────────────────────
def run_search_and_report(estimator, label, pkl_fname):
    search = RandomizedSearchCV(
        estimator, tree_param_grid,
        n_iter=20, cv=3,
        scoring='neg_root_mean_squared_error',
        random_state=SEED, n_jobs=1, refit=True,
        verbose=1,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_

    # CV R² (separate pass)
    cv_r2 = cross_val_score(best, X_train, y_train, cv=3, scoring='r2').mean()

    # Test metrics — log10_k
    y_pred_log  = best.predict(X_test)
    test_r2_log = r2_score(y_test, y_pred_log)

    # Back-transform to raw k
    y_test_raw   = 10 ** y_test
    y_pred_raw   = 10 ** y_pred_log
    test_r2_raw  = r2_score(y_test_raw, y_pred_raw)
    test_rmse_raw = np.sqrt(mean_squared_error(y_test_raw, y_pred_raw))
    test_mae_raw  = mean_absolute_error(y_test_raw, y_pred_raw)

    print("\n" + "=" * 55)
    print(f"{label} — Results")
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

    with open(f"{MODELS_DIR}/{pkl_fname}", 'wb') as f:
        pickle.dump(best, f)
    print(f"\nModel saved: {pkl_fname}")

    return {
        'label'       : label,
        'cv_r2'       : cv_r2,
        'test_r2_log' : test_r2_log,
        'test_r2_raw' : test_r2_raw,
        'test_rmse_raw': test_rmse_raw,
        'test_mae_raw' : test_mae_raw,
    }

# ── Random Forest Regressor ───────────────────────────────────
res_rf = run_search_and_report(
    RandomForestRegressor(random_state=SEED, n_jobs=-1),
    label     = "Random Forest Regressor",
    pkl_fname = "model_rf_s2.pkl",
)

# ── ExtraTrees Regressor ──────────────────────────────────────
res_et = run_search_and_report(
    ExtraTreesRegressor(random_state=SEED, n_jobs=-1),
    label     = "ExtraTrees Regressor",
    pkl_fname = "model_et_s2.pkl",
)

# ── Summary table ─────────────────────────────────────────────
print("\n" + "=" * 72)
print("SUMMARY — Stage 2 Regression (RF & ET)")
print("=" * 72)
print(f"{'Model':<26}  {'CV R²':>6}  {'Test R²(log)':>12}  "
      f"{'Test R²(raw)':>12}  {'RMSE(raw)':>9}  {'MAE(raw)':>8}")
print("-" * 72)
for r in [res_rf, res_et]:
    print(f"{r['label']:<26}  {r['cv_r2']:>6.4f}  {r['test_r2_log']:>12.4f}  "
          f"{r['test_r2_raw']:>12.4f}  {r['test_rmse_raw']:>9.4f}  {r['test_mae_raw']:>8.4f}")
print("\nNote: compare with XGBoost results from s2_task2_xgb.py")
