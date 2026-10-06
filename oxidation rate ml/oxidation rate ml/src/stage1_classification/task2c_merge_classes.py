import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import RandomOverSampler, SMOTE

from pathlib import Path
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
TARGET      = 'kinetics_class'
CLASS_NAMES = ['Linear', 'Parabolic', 'HigherOrder']
SEED        = 42

# Merge mapping: 2 → HigherOrder, 3 → HigherOrder
MERGE_MAP = {0: 0, 1: 1, 2: 2, 3: 2}

# ── Load data ────────────────────────────────────────────────
train = pd.read_csv(f"{DATA_DIR}/train_scaled.csv")
test  = pd.read_csv(f"{DATA_DIR}/test_scaled.csv")

feature_cols = [c for c in train.columns if c != TARGET]

# ── Class distribution before merging ────────────────────────
orig_label_map = {0: 'Linear', 1: 'Parabolic', 2: 'Cubic', 3: 'Quartic'}
print("=" * 55)
print("Class distribution BEFORE merging")
print("=" * 55)
for split_name, df in [('Train', train), ('Test', test)]:
    counts = df[TARGET].value_counts().sort_index()
    print(f"\n{split_name}:")
    for cls, cnt in counts.items():
        print(f"  {cls} ({orig_label_map[cls]:>9s}): {cnt}")

# ── Apply merge mapping ───────────────────────────────────────
train[TARGET] = train[TARGET].map(MERGE_MAP)
test[TARGET]  = test[TARGET].map(MERGE_MAP)

# ── Class distribution after merging ─────────────────────────
new_label_map = {0: 'Linear', 1: 'Parabolic', 2: 'HigherOrder'}
print("\n" + "=" * 55)
print("Class distribution AFTER merging")
print("=" * 55)
for split_name, df in [('Train', train), ('Test', test)]:
    counts = df[TARGET].value_counts().sort_index()
    print(f"\n{split_name}:")
    for cls, cnt in counts.items():
        print(f"  {cls} ({new_label_map[cls]:>11s}): {cnt}")

# ── Save 3-class datasets ─────────────────────────────────────
train.to_csv(f"{DATA_DIR}/train_scaled_3class.csv", index=False)
test.to_csv(f"{DATA_DIR}/test_scaled_3class.csv",   index=False)
print(f"\nSaved: train_scaled_3class.csv  |  test_scaled_3class.csv")

# ── Prepare arrays ────────────────────────────────────────────
X_train = train[feature_cols].values
y_train = train[TARGET].values
X_test  = test[feature_cols].values
y_test  = test[TARGET].values

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# ── Helper ────────────────────────────────────────────────────
def evaluate(label, model, X_tr, y_tr, X_te, y_te):
    cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv,
                                scoring='f1_macro', n_jobs=-1)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    acc    = accuracy_score(y_te, y_pred)
    mf1    = f1_score(y_te, y_pred, average='macro', zero_division=0)
    report = classification_report(
        y_te, y_pred, target_names=CLASS_NAMES,
        zero_division=0, output_dict=True
    )
    per_class_f1 = {cls: report[cls]['f1-score'] for cls in CLASS_NAMES}

    print(f"\n{'='*55}")
    print(label)
    print(f"{'='*55}")
    print(f"CV macro-F1 (train, 5-fold) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Test accuracy               : {acc:.4f}")
    print(f"Test macro-F1               : {mf1:.4f}")
    print("Per-class F1:")
    for cls in CLASS_NAMES:
        print(f"  {cls:>11s}: {per_class_f1[cls]:.4f}")

    return {
        'label'        : label,
        'cv_f1'        : cv_scores.mean(),
        'test_acc'     : acc,
        'test_mf1'     : mf1,
        'higherorder_f1': per_class_f1['HigherOrder'],
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
smote        = SMOTE(k_neighbors=2, random_state=SEED)
X_sm, y_sm   = smote.fit_resample(X_train, y_train)
vals, cnts   = np.unique(y_sm, return_counts=True)
print(f"\nClass distribution after SMOTE(k_neighbors=2): "
      f"{dict(zip(vals.tolist(), cnts.tolist()))}")

model_c = ExtraTreesClassifier(n_estimators=400, random_state=SEED)
res_c = evaluate("Strategy C — SMOTE(k_neighbors=2)",
                 model_c, X_sm, y_sm, X_test, y_test)

# ── Summary table ─────────────────────────────────────────────
print(f"\n{'='*72}")
print("SUMMARY  (winner selected by HigherOrder F1 — the merged minority class)")
print(f"{'='*72}")
print(f"{'Strategy':<40}  {'CV F1':>6}  {'Test F1':>7}  {'HigherOrder F1':>14}")
print("-" * 72)

results = [res_a, res_b, res_c]
best    = max(results, key=lambda r: r['higherorder_f1'])

for r in results:
    marker = " ← WINNER" if r['label'] == best['label'] else ""
    print(f"{r['label']:<40}  {r['cv_f1']:>6.4f}  {r['test_mf1']:>7.4f}  "
          f"{r['higherorder_f1']:>14.4f}{marker}")

print(f"\nWINNER         = '{best['label']}'")
print(f"HigherOrder F1 = {best['higherorder_f1']:.4f}")
