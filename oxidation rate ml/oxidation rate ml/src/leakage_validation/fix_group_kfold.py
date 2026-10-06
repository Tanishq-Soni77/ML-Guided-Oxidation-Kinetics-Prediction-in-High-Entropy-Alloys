"""
fix_group_kfold.py
==================
Fixes data leakage in hyperparameter tuning by replacing KFold with GroupKFold
so that same alloy group never appears in both train and validation folds.

Runs BEFORE (KFold) and AFTER (GroupKFold) back-to-back and prints a
side-by-side comparison table.

Stage 1 : ExtraTreesClassifier  — scored on f1_macro
Stage 2 : XGBoost Regressor     — scored on r2
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import (
    RandomizedSearchCV, KFold, GroupKFold
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    r2_score, mean_squared_error, mean_absolute_error
)
from xgboost import XGBRegressor

# ─────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────
from pathlib import Path
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SEED       = 42
CLASS_NAMES = ['Linear', 'Parabolic', 'Cubic', 'Quartic']

# ─────────────────────────────────────────────────────────────────
# Load Stage 1 data
# (pre-scaled features, alloy_group comes from the raw data file
#  which is row-aligned with the scaled file)
# ─────────────────────────────────────────────────────────────────
s1_train_sc  = pd.read_csv(f"{DATA_DIR}/train_scaled.csv")
s1_test_sc   = pd.read_csv(f"{DATA_DIR}/test_scaled.csv")
s1_train_raw = pd.read_csv(f"{DATA_DIR}/train_data.csv")
s1_test_raw  = pd.read_csv(f"{DATA_DIR}/test_data.csv")

S1_TARGET   = 'kinetics_class'
s1_feat_cols = [c for c in s1_train_sc.columns if c != S1_TARGET]

X1_train  = s1_train_sc[s1_feat_cols].values
y1_train  = s1_train_sc[S1_TARGET].values
X1_test   = s1_test_sc[s1_feat_cols].values
y1_test   = s1_test_sc[S1_TARGET].values
groups1   = s1_train_raw['alloy_group'].values   # group labels aligned row-for-row

sw1 = compute_sample_weight('balanced', y1_train)

print(f"Stage 1  — Train: {X1_train.shape}  Test: {X1_test.shape}")
print(f"  Unique alloy groups in train: {len(np.unique(groups1))}")

# ─────────────────────────────────────────────────────────────────
# Load Stage 2 data
# (scaled file already carries alloy_group column)
# ─────────────────────────────────────────────────────────────────
S2_TARGET    = 'log10_k'
S2_FEAT_COLS = [
    'Al', 'Si', 'Ti', 'V', 'Cr', 'Zr', 'Nb', 'Mo', 'Hf', 'Ta', 'W',
    'Tm', 'delta', 'To', 't',
    'protective_sum', 'risk_sum', 'Al_Cr_interaction', 'thermal_dose',
    'predicted_n',
]

s2_train = pd.read_csv(f"{DATA_DIR}/train_s2_scaled.csv")
s2_test  = pd.read_csv(f"{DATA_DIR}/test_s2_scaled.csv")

X2_train  = s2_train[S2_FEAT_COLS].values
y2_train  = s2_train[S2_TARGET].values
X2_test   = s2_test[S2_FEAT_COLS].values
y2_test   = s2_test[S2_TARGET].values
groups2   = s2_train['alloy_group'].values

print(f"\nStage 2  — Train: {X2_train.shape}  Test: {X2_test.shape}")
print(f"  Unique alloy groups in train: {len(np.unique(groups2))}")

# ─────────────────────────────────────────────────────────────────
# Hyperparameter search spaces  (unchanged from original scripts)
# ─────────────────────────────────────────────────────────────────
et_param_grid = {
    'n_estimators'    : [200, 400, 600],
    'max_depth'       : [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'class_weight'    : ['balanced', None],
}

xgb_param_grid = {
    'n_estimators'    : [300, 500, 700],
    'max_depth'       : [3, 4, 5],
    'learning_rate'   : [0.01, 0.03, 0.05, 0.1],
    'subsample'       : [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'reg_alpha'       : [0, 0.1, 0.5, 1.0],
    'reg_lambda'      : [1, 2, 5],
    'min_child_weight': [1, 3, 5],
}

# ─────────────────────────────────────────────────────────────────
# Helper: evaluate Stage 1 classifier on test set
# ─────────────────────────────────────────────────────────────────
def eval_s1(model, label=""):
    y_pred = model.predict(X1_test)
    acc    = accuracy_score(y1_test, y_pred)
    mf1    = f1_score(y1_test, y_pred, average='macro', zero_division=0)
    report = classification_report(
        y1_test, y_pred, target_names=CLASS_NAMES,
        zero_division=0, output_dict=True
    )
    per_class = {cls: report[cls]['f1-score'] for cls in CLASS_NAMES}
    if label:
        print(f"  Test accuracy  : {acc:.4f}")
        print(f"  Test macro-F1  : {mf1:.4f}")
        print(f"  Per-class F1:")
        for cls, v in per_class.items():
            print(f"    {cls:>12s}: {v:.4f}")
    return acc, mf1, per_class

# ─────────────────────────────────────────────────────────────────
# Helper: evaluate Stage 2 regressor on test set
# ─────────────────────────────────────────────────────────────────
def eval_s2(model, label=""):
    y_pred_log = model.predict(X2_test)
    y_raw      = 10 ** y2_test
    y_pred_raw = 10 ** y_pred_log
    r2_log  = r2_score(y2_test, y_pred_log)
    r2_raw  = r2_score(y_raw, y_pred_raw)
    rmse    = np.sqrt(mean_squared_error(y_raw, y_pred_raw))
    mae     = mean_absolute_error(y_raw, y_pred_raw)
    if label:
        print(f"  Test R²  (log10_k): {r2_log:.4f}")
        print(f"  Test R²  (raw k)  : {r2_raw:.4f}")
        print(f"  Test RMSE (raw k) : {rmse:.4f}")
        print(f"  Test MAE  (raw k) : {mae:.4f}")
    return r2_log, r2_raw, rmse, mae

SEP = "=" * 70

# ═════════════════════════════════════════════════════════════════
# BEFORE — KFold (original, leaky)
# ═════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("BEFORE  —  KFold (no group awareness, data leakage present)")
print(SEP)

# ── Stage 1 BEFORE ───────────────────────────────────────────────
print("\n[Stage 1] ExtraTreesClassifier  |  cv=KFold(5)  |  scoring=f1_macro")
et_base_b = ExtraTreesClassifier(random_state=SEED, n_jobs=-1)
search_s1_before = RandomizedSearchCV(
    et_base_b, et_param_grid,
    n_iter=30,
    cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
    scoring='f1_macro',
    random_state=SEED, n_jobs=-1, refit=True,
)
search_s1_before.fit(X1_train, y1_train, sample_weight=sw1)
print(f"  Best params : {search_s1_before.best_params_}")
print(f"  CV macro-F1 : {search_s1_before.best_score_:.4f}")
b_acc, b_mf1, b_pc = eval_s1(search_s1_before.best_estimator_, label="print")

# ── Stage 2 BEFORE ───────────────────────────────────────────────
print("\n[Stage 2] XGBoost Regressor  |  cv=KFold(5)  |  scoring=r2")
xgb_base_b = XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmse',
    verbosity=0, random_state=SEED, n_jobs=1,
)
search_s2_before = RandomizedSearchCV(
    xgb_base_b, xgb_param_grid,
    n_iter=20,
    cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
    scoring='r2',
    random_state=SEED, n_jobs=1, refit=True,
)
search_s2_before.fit(X2_train, y2_train)
print(f"  Best params : {search_s2_before.best_params_}")
print(f"  CV R²       : {search_s2_before.best_score_:.4f}")
b_r2log, b_r2raw, b_rmse, b_mae = eval_s2(search_s2_before.best_estimator_, label="print")

# ═════════════════════════════════════════════════════════════════
# AFTER — GroupKFold (fixed, no leakage)
# ═════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("AFTER   —  GroupKFold(5)  (alloy group never splits across folds)")
print(SEP)

# ── Stage 1 AFTER ────────────────────────────────────────────────
print("\n[Stage 1] ExtraTreesClassifier  |  cv=GroupKFold(5)  |  scoring=f1_macro")
et_base_a = ExtraTreesClassifier(random_state=SEED, n_jobs=-1)
search_s1_after = RandomizedSearchCV(
    et_base_a, et_param_grid,
    n_iter=30,
    cv=GroupKFold(n_splits=5),
    scoring='f1_macro',
    random_state=SEED, n_jobs=-1, refit=True,
)
search_s1_after.fit(X1_train, y1_train, sample_weight=sw1, groups=groups1)
print(f"  Best params : {search_s1_after.best_params_}")
print(f"  CV macro-F1 : {search_s1_after.best_score_:.4f}")
a_acc, a_mf1, a_pc = eval_s1(search_s1_after.best_estimator_, label="print")

# ── Stage 2 AFTER ────────────────────────────────────────────────
print("\n[Stage 2] XGBoost Regressor  |  cv=GroupKFold(5)  |  scoring=r2")
xgb_base_a = XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmse',
    verbosity=0, random_state=SEED, n_jobs=1,
)
search_s2_after = RandomizedSearchCV(
    xgb_base_a, xgb_param_grid,
    n_iter=20,
    cv=GroupKFold(n_splits=5),
    scoring='r2',
    random_state=SEED, n_jobs=1, refit=True,
)
search_s2_after.fit(X2_train, y2_train, groups=groups2)
print(f"  Best params : {search_s2_after.best_params_}")
print(f"  CV R²       : {search_s2_after.best_score_:.4f}")
a_r2log, a_r2raw, a_rmse, a_mae = eval_s2(search_s2_after.best_estimator_, label="print")

# ═════════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ═════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("BEFORE vs AFTER — Comparison Table")
print(SEP)

# Stage 1
print("\n── Stage 1: ExtraTreesClassifier ────────────────────────────────")
print(f"{'Metric':<22}  {'BEFORE (KFold)':>16}  {'AFTER (GroupKFold)':>18}  {'Δ':>8}")
print("-" * 70)
rows_s1 = [
    ("Test Accuracy",  b_acc,  a_acc),
    ("Test macro-F1",  b_mf1,  a_mf1),
]
for cls in CLASS_NAMES:
    rows_s1.append((f"F1 {cls}", b_pc[cls], a_pc[cls]))

for label, bv, av in rows_s1:
    delta = av - bv
    sign  = "+" if delta >= 0 else ""
    print(f"  {label:<20}  {bv:>16.4f}  {av:>18.4f}  {sign}{delta:>7.4f}")

# Stage 2
print("\n── Stage 2: XGBoost Regressor ───────────────────────────────────")
print(f"{'Metric':<22}  {'BEFORE (KFold)':>16}  {'AFTER (GroupKFold)':>18}  {'Δ':>8}")
print("-" * 70)
rows_s2 = [
    ("Test R² log10(k)",  b_r2log,  a_r2log),
    ("Test R² raw k",     b_r2raw,  a_r2raw),
    ("Test RMSE raw k",   b_rmse,   a_rmse),
    ("Test MAE  raw k",   b_mae,    a_mae),
]
for label, bv, av in rows_s2:
    delta = av - bv
    sign  = "+" if delta >= 0 else ""
    print(f"  {label:<20}  {bv:>16.4f}  {av:>18.4f}  {sign}{delta:>7.4f}")

print(f"\n{SEP}")
print("NOTE: Lower RMSE/MAE = better.  For RMSE/MAE, negative Δ is an improvement.")
print(SEP)
