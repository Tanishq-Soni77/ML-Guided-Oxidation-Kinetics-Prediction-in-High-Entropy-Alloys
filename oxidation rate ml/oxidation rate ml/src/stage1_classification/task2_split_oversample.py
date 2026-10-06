import re
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from xgboost import XGBClassifier

from pathlib import Path
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
df  = pd.read_csv(f"{DATA_DIR}/processed_data.csv")

# ─────────────────────────────────────────────────────────────
# TASK 1 — Parse alloy_group from reference
# ─────────────────────────────────────────────────────────────
print("=" * 65)
print("TASK 1 — Alloy group extraction")
print("=" * 65)

def parse_alloy_name(ref: str) -> str:
    matches = re.findall(r'\(([^()]+)\)', ref)
    if not matches:
        return ref.strip()
    content = matches[-1].strip()

    # Strip trailing descriptors
    for suffix in [' oxidation data', ' ternary oxidation data', ' equiatomic', ' variant']:
        if content.lower().endswith(suffix.lower()):
            content = content[: -len(suffix)].strip()
            break

    # "Al... at 1000°C, Linear kinetics" → drop temperature/kinetics suffix
    if ' at ' in content:
        content = content.split(' at ')[0].strip()

    # "label; composition" → take part after ";"
    if '; ' in content:
        parts = content.split('; ')
        content = parts[-1].strip()
        for suffix in [' oxidation data', ' variant', ' equiatomic']:
            if content.lower().endswith(suffix.lower()):
                content = content[: -len(suffix)].strip()
                break

    return content

df['alloy_group'] = df['reference'].apply(parse_alloy_name)

n_groups = df['alloy_group'].nunique()
print(f"\nNumber of unique alloy groups: {n_groups}")
print(f"\nalloy_group value_counts:\n{df['alloy_group'].value_counts().to_string()}")

# ─────────────────────────────────────────────────────────────
# TASK 2 — GroupShuffleSplit (group-aware train/test)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("TASK 2 — GroupShuffleSplit (no alloy group leakage)")
print("=" * 65)

gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
groups = df['alloy_group'].values
train_idx, test_idx = next(gss.split(df, df['kinetics_class'], groups=groups))

train_df = df.iloc[train_idx].copy()
test_df  = df.iloc[test_idx].copy()

train_groups = set(train_df['alloy_group'].unique())
test_groups  = set(test_df['alloy_group'].unique())
overlap      = train_groups & test_groups

print(f"\nTrain size: {len(train_df)}   |   Test size: {len(test_df)}")

print(f"\nUnique alloy groups in TRAIN ({len(train_groups)}):")
for g in sorted(train_groups):
    print(f"  {g}")
print(f"\nUnique alloy groups in TEST ({len(test_groups)}):")
for g in sorted(test_groups):
    print(f"  {g}")

if overlap:
    print(f"\n[ERROR] Overlap detected: {overlap}")
    raise SystemExit(1)
else:
    print("\n[OK] Zero overlap confirmed.")

print(f"\nkinetics_class distribution — TRAIN:\n{train_df['kinetics_class'].value_counts().sort_index().to_string()}")
print(f"\nkinetics_class distribution — TEST:\n{test_df['kinetics_class'].value_counts().sort_index().to_string()}")

train_df.to_csv(f"{DATA_DIR}/train_data.csv", index=False)
test_df.to_csv(f"{DATA_DIR}/test_data.csv",   index=False)
print("\nSaved: train_data.csv  |  test_data.csv  [FROZEN]")

# ─────────────────────────────────────────────────────────────
# TASK 3 — Oversampling strategy comparison (train set only)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("TASK 3 — Oversampling strategy comparison")
print("=" * 65)

FEATURE_COLS = [
    'Al','Si','Ti','V','Cr','Zr','Nb','Mo','Hf','Ta','W',
    'Tm','delta','To','t',
    'protective_sum','risk_sum','Al_Cr_interaction','thermal_dose',
]
TARGET = 'kinetics_class'

X_tr = train_df[FEATURE_COLS].values
y_tr = train_df[TARGET].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

label_map = {0: 'Linear', 1: 'Parabolic', 2: 'Cubic', 3: 'Quartic'}

def class_dist_str(y):
    vals, cnts = np.unique(y, return_counts=True)
    return "  " + " | ".join(f"class {v}({label_map.get(v,'?')}): {c}" for v, c in zip(vals, cnts))

def run_cv_manual(X, y, sampler=None, use_sample_weight=False):
    """5-fold CV with optional per-fold resampling / sample weighting."""
    fold_f1 = []
    for tr_idx, val_idx in skf.split(X, y):
        Xtr, Xval = X[tr_idx], X[val_idx]
        ytr, yval = y[tr_idx], y[val_idx]
        sw = None

        if sampler is not None:
            mc = int(pd.Series(ytr).value_counts().min())
            kk = min(5, mc - 1)
            sampler.set_params(k_neighbors=kk)
            Xtr, ytr = sampler.fit_resample(Xtr, ytr)

        if use_sample_weight:
            sw = compute_sample_weight('balanced', ytr)

        model = XGBClassifier(n_estimators=100, random_state=42,
                               eval_metric='mlogloss', verbosity=0)
        model.fit(Xtr, ytr, sample_weight=sw)
        yp = model.predict(Xval)
        fold_f1.append(f1_score(yval, yp, average='macro', zero_division=0))
    return np.array(fold_f1)

# Min class count determines SMOTE eligibility
class_counts = pd.Series(y_tr).value_counts()
min_count    = int(class_counts.min())
smote_ok     = min_count >= 6

print(f"\nClass counts in train: {dict(class_counts.sort_index())}")
print(f"Min class count: {min_count}  →  SMOTE eligible: {smote_ok}")

results = {}

# ── A: No oversampling ────────────────────────────────────────
print("\n--- Strategy A: No oversampling ---")
print(f"Class distribution:\n{class_dist_str(y_tr)}")
scores_a = run_cv_manual(X_tr, y_tr)
results['A_NoOversampling'] = scores_a.mean()
print(f"CV macro-F1: {scores_a.mean():.4f} ± {scores_a.std():.4f}")

# ── B: SMOTE ─────────────────────────────────────────────────
print("\n--- Strategy B: SMOTE ---")
if not smote_ok:
    print(f"SKIPPED — any class < 6 samples (min={min_count})")
    results['B_SMOTE'] = np.nan
else:
    k_init = min(5, min_count - 1)
    smote_sampler = SMOTE(k_neighbors=k_init, random_state=42)
    # Show distribution after full-train resample for reference
    X_sm, y_sm = SMOTE(k_neighbors=k_init, random_state=42).fit_resample(X_tr, y_tr)
    print(f"Class distribution after SMOTE (full train):\n{class_dist_str(y_sm)}")
    scores_b = run_cv_manual(X_tr, y_tr, sampler=smote_sampler)
    results['B_SMOTE'] = scores_b.mean()
    print(f"CV macro-F1: {scores_b.mean():.4f} ± {scores_b.std():.4f}")

# ── C: BorderlineSMOTE ────────────────────────────────────────
print("\n--- Strategy C: BorderlineSMOTE ---")
if not smote_ok:
    print(f"SKIPPED — any class < 6 samples (min={min_count})")
    results['C_BorderlineSMOTE'] = np.nan
else:
    k_init = min(5, min_count - 1)
    bl_sampler = BorderlineSMOTE(k_neighbors=k_init, random_state=42)
    X_bl, y_bl = BorderlineSMOTE(k_neighbors=k_init, random_state=42).fit_resample(X_tr, y_tr)
    print(f"Class distribution after BorderlineSMOTE (full train):\n{class_dist_str(y_bl)}")
    scores_c = run_cv_manual(X_tr, y_tr, sampler=bl_sampler)
    results['C_BorderlineSMOTE'] = scores_c.mean()
    print(f"CV macro-F1: {scores_c.mean():.4f} ± {scores_c.std():.4f}")

# ── D: class_weight='balanced' ────────────────────────────────
print("\n--- Strategy D: class_weight='balanced' in model ---")
print(f"Class distribution (unchanged):\n{class_dist_str(y_tr)}")
scores_d = run_cv_manual(X_tr, y_tr, use_sample_weight=True)
results['D_ClassWeightBalanced'] = scores_d.mean()
print(f"CV macro-F1: {scores_d.mean():.4f} ± {scores_d.std():.4f}")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY — CV macro-F1 by strategy")
print("=" * 65)
valid_results = {k: v for k, v in results.items() if not np.isnan(v)}
best_score    = max(valid_results.values())

for name, score in sorted(results.items()):
    if np.isnan(score):
        print(f"  {name:30s}: SKIPPED")
    else:
        marker = " ← WINNER" if score == best_score else ""
        print(f"  {name:30s}: {score:.4f}{marker}")

WINNER_STRATEGY = max(valid_results, key=valid_results.get)
print(f"\nWINNER_STRATEGY = '{WINNER_STRATEGY}'")
