import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.preprocessing import RobustScaler

from pathlib import Path
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
TARGET = 'log10_k'

FEATURE_COLS = [
    'Al', 'Si', 'Ti', 'V', 'Cr', 'Zr', 'Nb', 'Mo', 'Hf', 'Ta', 'W',
    'Tm', 'delta', 'To', 't',
    'protective_sum', 'risk_sum', 'Al_Cr_interaction', 'thermal_dose',
    'predicted_n',
]

EXTRA_COLS = ['log10_k', 'alloy_group', 'kinetics_class']

# ── Load ─────────────────────────────────────────────────────
train = pd.read_csv(f"{DATA_DIR}/train_data_s2.csv")
test  = pd.read_csv(f"{DATA_DIR}/test_data_s2.csv")

# Normalise predicted_n column name (train → predicted_n_train, test → predicted_n_test)
train = train.rename(columns={'predicted_n_train': 'predicted_n'})
test  = test.rename(columns={'predicted_n_test':  'predicted_n'})

print(f"Train shape (raw): {train.shape}")
print(f"Test  shape (raw): {test.shape}")

# ── Feature confirmation ──────────────────────────────────────
missing_train = [c for c in FEATURE_COLS + EXTRA_COLS if c not in train.columns]
missing_test  = [c for c in FEATURE_COLS + EXTRA_COLS if c not in test.columns]
if missing_train:
    raise ValueError(f"Missing columns in train: {missing_train}")
if missing_test:
    raise ValueError(f"Missing columns in test: {missing_test}")

print(f"\nFeature list confirmed ({len(FEATURE_COLS)} features):")
for col in FEATURE_COLS:
    print(f"  {col}")

# ── log10_k stats before scaling ─────────────────────────────
print(f"\nTrain  log10_k stats:")
print(f"  mean={train[TARGET].mean():.4f}  min={train[TARGET].min():.4f}"
      f"  max={train[TARGET].max():.4f}  std={train[TARGET].std():.4f}")
print(f"Test   log10_k stats:")
print(f"  mean={test[TARGET].mean():.4f}  min={test[TARGET].min():.4f}"
      f"  max={test[TARGET].max():.4f}  std={test[TARGET].std():.4f}")

# ── RobustScaler — fit on train only ─────────────────────────
scaler  = RobustScaler()
X_train = scaler.fit_transform(train[FEATURE_COLS])
X_test  = scaler.transform(test[FEATURE_COLS])

# ── Assemble output DataFrames ────────────────────────────────
train_scaled = pd.DataFrame(X_train, columns=FEATURE_COLS)
for col in EXTRA_COLS:
    train_scaled[col] = train[col].values

test_scaled = pd.DataFrame(X_test, columns=FEATURE_COLS)
for col in EXTRA_COLS:
    test_scaled[col] = test[col].values

# ── Save ─────────────────────────────────────────────────────
train_scaled.to_csv(f"{DATA_DIR}/train_s2_scaled.csv", index=False)
test_scaled.to_csv(f"{DATA_DIR}/test_s2_scaled.csv",   index=False)

print(f"\nSaved: train_s2_scaled.csv  shape={train_scaled.shape}")
print(f"Saved: test_s2_scaled.csv   shape={test_scaled.shape}")
