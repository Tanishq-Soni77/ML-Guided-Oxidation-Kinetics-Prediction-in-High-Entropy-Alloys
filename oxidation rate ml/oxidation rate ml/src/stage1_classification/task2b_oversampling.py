import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report
)
from imblearn.over_sampling import RandomOverSampler, SMOTE

from pathlib import Path
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
TARGET      = 'kinetics_class'
CLASS_NAMES = ['Linear', 'Parabolic', 'Cubic', 'Quartic']
CUBIC_IDX   = 2          # class label for Cubic
SEED        = 42

# ── Load data ────────────────────────────────────────────────
train = pd.read_csv(f"{DATA_DIR}/train_scaled.csv")
test  = pd.read_csv(f"{DATA_DIR}/test_scaled.csv")

feature_cols = [c for c in train.columns if c != TARGET]

X_train = train[feature_cols].values
y_train = train[TARGET].values
X_test  = test[feature_cols].values
y_test  = test[TARGET].values

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"Train class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# ── Helper ────────────────────────────────────────────────────
def evaluate(label, model, X_tr, y_tr, X_te, y_te):
    """CV on (X_tr, y_tr), final eval on (X_te, y_te)."""
    cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv,
                                scoring='f1_macro', n_jobs=-1)

    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    acc  = accuracy_score(y_te, y_pred)
    mf1  = f1_score(y_te, y_pred, average='macro', zero_division=0)
    report = classification_report(
        y_te, y_pred, target_names=CLASS_NAMES,
        zero_division=0, output_dict=True
    )
    per_class_f1 = {cls: report[cls]['f1-score'] for cls in CLASS_NAMES}
    cubic_f1     = per_class_f1['Cubic']

    print(f"\n{'='*55}")
    print(f"{label}")
    print(f"{'='*55}")
    print(f"CV macro-F1 (train, 5-fold) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Test accuracy               : {acc:.4f}")
    print(f"Test macro-F1               : {mf1:.4f}")
    print("Per-class F1:")
    for cls in CLASS_NAMES:
        print(f"  {cls:>10s}: {per_class_f1[cls]:.4f}")

    return {
        'label'    : label,
        'cv_f1'    : cv_scores.mean(),
        'test_acc' : acc,
        'test_mf1' : mf1,
        'cubic_f1' : cubic_f1,
        **{f'f1_{cls}': per_class_f1[cls] for cls in CLASS_NAMES},
    }

# ── Strategy A: class_weight='balanced', no oversampling ──────
model_a = ExtraTreesClassifier(
    n_estimators=400, class_weight='balanced', random_state=SEED
)
res_a = evaluate("Strategy A — class_weight='balanced' (no oversampling)",
                 model_a, X_train, y_train, X_test, y_test)

# ── Strategy B: RandomOverSampler ─────────────────────────────
ros          = RandomOverSampler(random_state=SEED)
X_ros, y_ros = ros.fit_resample(X_train, y_train)
vals, cnts   = np.unique(y_ros, return_counts=True)
print(f"\nClass distribution after RandomOverSampler: "
      f"{dict(zip(vals.tolist(), cnts.tolist()))}")

model_b = ExtraTreesClassifier(n_estimators=400, random_state=SEED)
res_b = evaluate("Strategy B — RandomOverSampler",
                 model_b, X_ros, y_ros, X_test, y_test)

# ── Strategy C: SMOTE(k_neighbors=2) ─────────────────────────
smote          = SMOTE(k_neighbors=2, random_state=SEED)
X_sm, y_sm     = smote.fit_resample(X_train, y_train)
vals, cnts     = np.unique(y_sm, return_counts=True)
print(f"\nClass distribution after SMOTE(k_neighbors=2): "
      f"{dict(zip(vals.tolist(), cnts.tolist()))}")

model_c = ExtraTreesClassifier(n_estimators=400, random_state=SEED)
res_c = evaluate("Strategy C — SMOTE(k_neighbors=2)",
                 model_c, X_sm, y_sm, X_test, y_test)

# ── Summary table ─────────────────────────────────────────────
print(f"\n{'='*70}")
print("SUMMARY  (winner selected by Cubic F1 — the problem class)")
print(f"{'='*70}")
header = f"{'Strategy':<40}  {'CV F1':>6}  {'Test F1':>7}  {'Cubic F1':>8}"
print(header)
print("-" * 70)

results = [res_a, res_b, res_c]
best    = max(results, key=lambda r: r['cubic_f1'])

for r in results:
    marker = " ← WINNER" if r['label'] == best['label'] else ""
    print(f"{r['label']:<40}  {r['cv_f1']:>6.4f}  {r['test_mf1']:>7.4f}  "
          f"{r['cubic_f1']:>8.4f}{marker}")

print(f"\nWINNER  = '{best['label']}'")
print(f"Cubic F1 = {best['cubic_f1']:.4f}")
