import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score

from pathlib import Path
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
TARGET = 'kinetics_class'
SEED   = 42

CLASS_NAMES = ['Linear', 'Parabolic', 'HigherOrder']
N_MAP       = {0: 1.0, 1: 2.0, 2: 3.0}

BEST_PARAMS = dict(
    n_estimators      = 600,
    min_samples_split = 5,
    max_depth         = None,
    class_weight      = 'balanced',
    random_state      = SEED,
)

# ── Load scaled data (for modelling) ─────────────────────────
train_sc = pd.read_csv(f"{DATA_DIR}/train_scaled_3class.csv")
test_sc  = pd.read_csv(f"{DATA_DIR}/test_scaled_3class.csv")

feature_cols = [c for c in train_sc.columns if c != TARGET]

X_train = train_sc[feature_cols].values
y_train = train_sc[TARGET].values
X_test  = test_sc[feature_cols].values
y_test  = test_sc[TARGET].values

# ── Load original data (to attach predictions) ────────────────
train_orig = pd.read_csv(f"{DATA_DIR}/train_data.csv")
test_orig  = pd.read_csv(f"{DATA_DIR}/test_data.csv")

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"Best params: {BEST_PARAMS}")

# ─────────────────────────────────────────────────────────────
# Step 1 — OOF predictions (5-fold KFold)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Step 1 — OOF predictions (5-fold KFold)")
print("=" * 55)

kf        = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof_preds = np.full(len(y_train), -1, dtype=int)

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
    Xtr, Xval = X_train[tr_idx], X_train[val_idx]
    ytr       = y_train[tr_idx]

    model = ExtraTreesClassifier(**BEST_PARAMS, n_jobs=-1)
    model.fit(Xtr, ytr)
    oof_preds[val_idx] = model.predict(Xval)

    fold_acc = accuracy_score(y_train[val_idx], oof_preds[val_idx])
    fold_f1  = f1_score(y_train[val_idx], oof_preds[val_idx],
                        average='macro', zero_division=0)
    print(f"  Fold {fold+1}: val_size={len(val_idx)}, "
          f"acc={fold_acc:.4f}, macro-F1={fold_f1:.4f}")

oof_acc = accuracy_score(y_train, oof_preds)
oof_f1  = f1_score(y_train, oof_preds, average='macro', zero_division=0)
print(f"\nOOF train accuracy : {oof_acc:.4f}")
print(f"OOF train macro-F1 : {oof_f1:.4f}")

# ─────────────────────────────────────────────────────────────
# Step 2 — Full train → test prediction
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Step 2 — Full train fit, test prediction")
print("=" * 55)

final_model = ExtraTreesClassifier(**BEST_PARAMS, n_jobs=-1)
final_model.fit(X_train, y_train)
test_preds = final_model.predict(X_test)

test_acc = accuracy_score(y_test, test_preds)
test_f1  = f1_score(y_test, test_preds, average='macro', zero_division=0)
print(f"Test accuracy  : {test_acc:.4f}")
print(f"Test macro-F1  : {test_f1:.4f}")

# ─────────────────────────────────────────────────────────────
# Step 3 — Map class → n value
# ─────────────────────────────────────────────────────────────
predicted_n_train = np.array([N_MAP[c] for c in oof_preds])
predicted_n_test  = np.array([N_MAP[c] for c in test_preds])

print(f"\nStep 3 — n-value mapping: {N_MAP}")
print(f"  Train n values: {np.unique(predicted_n_train, return_counts=True)}")
print(f"  Test  n values: {np.unique(predicted_n_test,  return_counts=True)}")

# ─────────────────────────────────────────────────────────────
# Step 4 — Attach to original data and save
# ─────────────────────────────────────────────────────────────
train_orig['predicted_n_train'] = predicted_n_train
test_orig['predicted_n_test']   = predicted_n_test

train_orig.to_csv(f"{DATA_DIR}/train_data_s2.csv", index=False)
test_orig.to_csv(f"{DATA_DIR}/test_data_s2.csv",   index=False)
print(f"\nSaved: train_data_s2.csv  |  test_data_s2.csv")

# ─────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("FINAL SUMMARY")
print("=" * 55)
print(f"Winning model        : ExtraTreesClassifier")
print(f"Hyperparameters      : {BEST_PARAMS}")
print(f"OOF train accuracy   : {oof_acc:.4f}")
print(f"OOF train macro-F1   : {oof_f1:.4f}")
print(f"Test accuracy        : {test_acc:.4f}")
print(f"Test macro-F1        : {test_f1:.4f}")
print(f"n-value mapping      : {N_MAP}  (HigherOrder → 3.0 as representative)")
