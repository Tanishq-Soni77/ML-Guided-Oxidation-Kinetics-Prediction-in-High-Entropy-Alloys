"""
Stage 1 — Model Selection
  - RobustScaler (fit on train, transform both)
  - Winning oversampling: D_ClassWeightBalanced (via sample_weight)
  - RandomizedSearchCV (cv=5, n_iter=30, f1_macro) for 3 classifiers
  - Evaluation: accuracy, macro-F1, per-class F1, confusion matrix
  - OOF predictions on train + test predictions from best model
  - Save train_data_s2.csv and test_data_s2.csv
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings, os
warnings.filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, KFold, cross_val_predict
)
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier

from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR  = _ROOT / "data"
PLOTS_DIR = _ROOT / "results" / "plots"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
SEED  = 42
CLASS_NAMES = ['Linear', 'Parabolic', 'Cubic', 'Quartic']
N_MAP = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}

FEATURE_COLS = [
    'Al','Si','Ti','V','Cr','Zr','Nb','Mo','Hf','Ta','W',
    'Tm','delta','To','t',
    'protective_sum','risk_sum','Al_Cr_interaction','thermal_dose',
]
TARGET = 'kinetics_class'

# ─────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────
train_df = pd.read_csv(f"{DATA_DIR}/train_data.csv")
test_df  = pd.read_csv(f"{DATA_DIR}/test_data.csv")

X_train_raw = train_df[FEATURE_COLS].values
y_train     = train_df[TARGET].values
X_test_raw  = test_df[FEATURE_COLS].values
y_test      = test_df[TARGET].values

print(f"Train: {X_train_raw.shape}  |  Test: {X_test_raw.shape}")

# ─────────────────────────────────────────────────────────────
# RobustScaler — fit on train only
# ─────────────────────────────────────────────────────────────
scaler    = RobustScaler()
X_train   = scaler.fit_transform(X_train_raw)
X_test    = scaler.transform(X_test_raw)
print("RobustScaler fitted on train, applied to both splits.")

# ─────────────────────────────────────────────────────────────
# Winning strategy: D_ClassWeightBalanced
# sample_weight computed from y_train; passed via fit_params
# ─────────────────────────────────────────────────────────────
sample_weights = compute_sample_weight('balanced', y_train)
print(f"Sample weights range: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def save_confusion_matrix(y_true, y_pred, title, fname):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{fname}", dpi=150)
    plt.close()
    print(f"  Saved: {fname}")

def evaluate(model, X_test, y_test, label):
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    mf1    = f1_score(y_test, y_pred, average='macro', zero_division=0)
    print(f"\n  Test Accuracy : {acc:.4f}")
    print(f"  Test macro-F1 : {mf1:.4f}")
    print(f"  Per-class F1  :")
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES,
                                   zero_division=0, output_dict=True)
    for cls in CLASS_NAMES:
        print(f"    {cls:>10s}: {report[cls]['f1-score']:.4f}")
    save_confusion_matrix(y_test, y_pred, f"{label} — Confusion Matrix",
                          f"cm_{label.lower().replace(' ','_')}.png")
    return acc, mf1, y_pred

sep = "=" * 65

# ─────────────────────────────────────────────────────────────
# Model 1 — XGBoost
# ─────────────────────────────────────────────────────────────
print(f"\n{sep}")
print("MODEL 1 — XGBoost Classifier")
print(sep)

xgb_param_grid = {
    'n_estimators'    : [200, 300, 400, 500],
    'max_depth'       : [3, 4, 5, 6],
    'learning_rate'   : [0.01, 0.05, 0.1, 0.15],
    'subsample'       : [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
}
xgb_base = XGBClassifier(
    eval_metric='mlogloss', verbosity=0, random_state=SEED, n_jobs=-1
)
xgb_search = RandomizedSearchCV(
    xgb_base, xgb_param_grid,
    n_iter=30, cv=5, scoring='f1_macro',
    random_state=SEED, n_jobs=-1, refit=True
)
xgb_search.fit(X_train, y_train, sample_weight=sample_weights)

print(f"  Best params: {xgb_search.best_params_}")
print(f"  CV macro-F1 (train): {xgb_search.best_score_:.4f}")
acc1, mf1_1, _ = evaluate(xgb_search.best_estimator_, X_test, y_test, "XGBoost")

# ─────────────────────────────────────────────────────────────
# Model 2 — Random Forest
# ─────────────────────────────────────────────────────────────
print(f"\n{sep}")
print("MODEL 2 — Random Forest Classifier")
print(sep)

rf_param_grid = {
    'n_estimators'   : [200, 400, 600],
    'max_depth'      : [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'class_weight'   : ['balanced', None],
}
rf_base = RandomForestClassifier(random_state=SEED, n_jobs=-1)
rf_search = RandomizedSearchCV(
    rf_base, rf_param_grid,
    n_iter=30, cv=5, scoring='f1_macro',
    random_state=SEED, n_jobs=-1, refit=True
)
rf_search.fit(X_train, y_train, sample_weight=sample_weights)

print(f"  Best params: {rf_search.best_params_}")
print(f"  CV macro-F1 (train): {rf_search.best_score_:.4f}")
acc2, mf1_2, _ = evaluate(rf_search.best_estimator_, X_test, y_test, "RandomForest")

# ─────────────────────────────────────────────────────────────
# Model 3 — ExtraTrees
# ─────────────────────────────────────────────────────────────
print(f"\n{sep}")
print("MODEL 3 — ExtraTreesClassifier (thesis baseline)")
print(sep)

et_param_grid = {
    'n_estimators'   : [200, 400, 600],
    'max_depth'      : [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'class_weight'   : ['balanced', None],
}
et_base = ExtraTreesClassifier(random_state=SEED, n_jobs=-1)
et_search = RandomizedSearchCV(
    et_base, et_param_grid,
    n_iter=30, cv=5, scoring='f1_macro',
    random_state=SEED, n_jobs=-1, refit=True
)
et_search.fit(X_train, y_train, sample_weight=sample_weights)

print(f"  Best params: {et_search.best_params_}")
print(f"  CV macro-F1 (train): {et_search.best_score_:.4f}")
acc3, mf1_3, _ = evaluate(et_search.best_estimator_, X_test, y_test, "ExtraTrees")

# ─────────────────────────────────────────────────────────────
# Stage 1 Summary — select best by test macro-F1
# ─────────────────────────────────────────────────────────────
print(f"\n{sep}")
print("STAGE 1 SUMMARY")
print(sep)
summary = {
    'XGBoost'    : (mf1_1, acc1, xgb_search.best_estimator_),
    'RandomForest': (mf1_2, acc2, rf_search.best_estimator_),
    'ExtraTrees' : (mf1_3, acc3, et_search.best_estimator_),
}
best_name = max(summary, key=lambda k: summary[k][0])
print(f"{'Model':<15}  {'Test Acc':>10}  {'Test macro-F1':>13}")
print("-" * 44)
for name, (mf1, acc, _) in summary.items():
    marker = " ← BEST" if name == best_name else ""
    print(f"{name:<15}  {acc:>10.4f}  {mf1:>13.4f}{marker}")

best_model = summary[best_name][2]
print(f"\nBest Stage 1 model: {best_name}")

# ─────────────────────────────────────────────────────────────
# OOF predictions on train (KFold, shuffle=True)
# ─────────────────────────────────────────────────────────────
print(f"\n{sep}")
print("OOF Predictions (5-fold KFold, shuffle=True, random_state=42)")
print(sep)

kf          = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof_preds   = np.full(len(y_train), -1, dtype=int)

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
    Xtr, Xval = X_train[tr_idx], X_train[val_idx]
    ytr       = y_train[tr_idx]
    sw_tr     = compute_sample_weight('balanced', ytr)

    # Re-instantiate best model type with best params (no data leakage)
    if best_name == 'XGBoost':
        m = XGBClassifier(**xgb_search.best_params_,
                           eval_metric='mlogloss', verbosity=0,
                           random_state=SEED, n_jobs=-1)
        m.fit(Xtr, ytr, sample_weight=sw_tr)
    elif best_name == 'RandomForest':
        m = RandomForestClassifier(**rf_search.best_params_,
                                    random_state=SEED, n_jobs=-1)
        m.fit(Xtr, ytr, sample_weight=sw_tr)
    else:
        m = ExtraTreesClassifier(**et_search.best_params_,
                                  random_state=SEED, n_jobs=-1)
        m.fit(Xtr, ytr, sample_weight=sw_tr)

    oof_preds[val_idx] = m.predict(Xval)
    print(f"  Fold {fold+1}: val size={len(val_idx)}, "
          f"macro-F1={f1_score(y_train[val_idx], oof_preds[val_idx], average='macro', zero_division=0):.4f}")

oof_acc = accuracy_score(y_train, oof_preds)
oof_f1  = f1_score(y_train, oof_preds, average='macro', zero_division=0)
print(f"\nOOF Train Accuracy : {oof_acc:.4f}")
print(f"OOF Train macro-F1 : {oof_f1:.4f}")

# Full-train prediction on test
# Retrain best model on all train data
print("\nRetraining best model on full train set for test prediction…")
sw_full = compute_sample_weight('balanced', y_train)
if best_name == 'XGBoost':
    final_model = XGBClassifier(**xgb_search.best_params_,
                                 eval_metric='mlogloss', verbosity=0,
                                 random_state=SEED, n_jobs=-1)
    final_model.fit(X_train, y_train, sample_weight=sw_full)
elif best_name == 'RandomForest':
    final_model = RandomForestClassifier(**rf_search.best_params_,
                                          random_state=SEED, n_jobs=-1)
    final_model.fit(X_train, y_train, sample_weight=sw_full)
else:
    final_model = ExtraTreesClassifier(**et_search.best_params_,
                                        random_state=SEED, n_jobs=-1)
    final_model.fit(X_train, y_train, sample_weight=sw_full)

test_preds = final_model.predict(X_test)
test_acc   = accuracy_score(y_test, test_preds)
print(f"Test Accuracy (full-train model): {test_acc:.4f}")

# ─────────────────────────────────────────────────────────────
# Map class → n value and save S2 files
# ─────────────────────────────────────────────────────────────
train_df_s2 = train_df.copy()
test_df_s2  = test_df.copy()

train_df_s2['predicted_n_train'] = [N_MAP[c] for c in oof_preds]
test_df_s2['predicted_n_test']   = [N_MAP[c] for c in test_preds]

train_df_s2.to_csv(f"{DATA_DIR}/train_data_s2.csv", index=False)
test_df_s2.to_csv(f"{DATA_DIR}/test_data_s2.csv",   index=False)

print(f"\nSaved train_data_s2.csv  |  test_data_s2.csv")
print(f"\nOOF train accuracy : {oof_acc:.4f}")
print(f"Test accuracy      : {test_acc:.4f}")
print(f"\nBEST_STAGE1_MODEL = '{best_name}'")
