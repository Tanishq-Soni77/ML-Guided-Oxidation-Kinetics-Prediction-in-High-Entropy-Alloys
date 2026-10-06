import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import r2_score

from pathlib import Path
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42

BASE_FEATURES = [
    'Al', 'Si', 'Ti', 'V', 'Cr', 'Zr', 'Nb', 'Mo', 'Hf', 'Ta', 'W',
    'Tm', 'delta', 'To', 't',
]

# Hardcoded results from our group-aware pipeline
OUR_S1_MACRO_F1  = 0.7531
OUR_S2_R2_LOG    = 0.7854
OUR_S2_R2_RAW    = 0.4260

# ── Load data ────────────────────────────────────────────────
df = pd.read_csv(f"{DATA_DIR}/processed_data.csv")

# Drop rows with missing targets
df = df.dropna(subset=['n', 'k'])

X        = df[BASE_FEATURES].values
y_n      = df['n'].values
y_log10k = np.log10(df['k'].values)      # Stage 2 target in log scale

print(f"Dataset shape: {df.shape}")
print(f"Features used: {BASE_FEATURES}")

# ─────────────────────────────────────────────────────────────
# Step 1 — Random 80/20 split (NO GroupShuffleSplit)
# ─────────────────────────────────────────────────────────────
(X_train, X_test,
 y_n_train, y_n_test,
 y_log_train, y_log_test) = train_test_split(
    X, y_n, y_log10k,
    test_size=0.20, random_state=SEED
)

print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")
print("Split: random 80/20 (NO GroupShuffleSplit — leakage baseline)")

# ─────────────────────────────────────────────────────────────
# Step 2 — Stage 1: Predict n as continuous regression
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Stage 1 — Predict n (ExtraTreesRegressor)")
print("=" * 55)

etr = ExtraTreesRegressor(n_estimators=400, random_state=SEED, n_jobs=-1)
etr.fit(X_train, y_n_train)
n_pred_train = etr.predict(X_train)
n_pred_test  = etr.predict(X_test)

s1_r2 = r2_score(y_n_test, n_pred_test)
print(f"Stage 1 R² (n, raw scale): {s1_r2:.4f}")

# ─────────────────────────────────────────────────────────────
# Step 3 — Stage 2: predicted_n as feature, predict log10(k)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Stage 2 — Predict log10(k) (RandomForestRegressor)")
print("=" * 55)

X_train_s2 = np.column_stack([X_train, n_pred_train])
X_test_s2  = np.column_stack([X_test,  n_pred_test])

rfr = RandomForestRegressor(n_estimators=400, random_state=SEED, n_jobs=-1)
rfr.fit(X_train_s2, y_log_train)
log_pred_test = rfr.predict(X_test_s2)

s2_r2_log = r2_score(y_log_test, log_pred_test)

actual_k    = 10 ** y_log_test
predicted_k = 10 ** log_pred_test
s2_r2_raw   = r2_score(actual_k, predicted_k)

print(f"Stage 2 R² (log10_k) : {s2_r2_log:.4f}")
print(f"Stage 2 R² (raw k)   : {s2_r2_raw:.4f}")

leakage_inflation = s2_r2_raw - OUR_S2_R2_RAW

# ─────────────────────────────────────────────────────────────
# Step 4 — Comparison table
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("LEAKAGE COMPARISON TABLE")
print("=" * 75)

col_w = [36, 16, 18, 16]
header = (f"{'Protocol':<{col_w[0]}}  {'Stage 1 Metric':>{col_w[1]}}"
          f"  {'Stage 2 R²log(k)':>{col_w[2]}}  {'Stage 2 R²raw k':>{col_w[3]}}")
divider = "-" * 75
print(header)
print(divider)

thesis_row = (
    f"{'Thesis: random split, ETR regress':<{col_w[0]}}"
    f"  {'R²='+f'{s1_r2:.4f}':>{col_w[1]}}"
    f"  {s2_r2_log:>{col_w[2]}.4f}"
    f"  {s2_r2_raw:>{col_w[3]}.4f}"
)
our_row = (
    f"{'Ours: GroupSplit, XGB classify':<{col_w[0]}}"
    f"  {'macro-F1='+f'{OUR_S1_MACRO_F1}':>{col_w[1]}}"
    f"  {OUR_S2_R2_LOG:>{col_w[2]}.4f}"
    f"  {OUR_S2_R2_RAW:>{col_w[3]}.4f}"
)
print(thesis_row)
print(our_row)
print(divider)

print(f"\nLeakage inflation = {leakage_inflation:+.4f} R² units "
      f"(thesis Stage 2 R² raw k {s2_r2_raw:.4f} − ours {OUR_S2_R2_RAW:.4f})")

if leakage_inflation > 0:
    print("=> Random split inflates Stage 2 performance due to alloy group leakage.")
else:
    print("=> No inflation detected; random split did not outperform group-aware split.")
