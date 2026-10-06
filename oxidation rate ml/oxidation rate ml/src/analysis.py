import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _ROOT / "data"
PLOTS_DIR = _ROOT / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# NOTE: the raw literature-mined dataset used for this initial EDA script is
# not included in this repo (see README > Data). Place your source CSV at
# data/raw_literature_dataset.csv to reproduce this script from scratch;
# data/processed_data.csv (its cleaned output) is already provided for the
# downstream pipeline stages.
DATA_PATH = DATA_DIR / "raw_literature_dataset.csv"
OUT_DIR   = PLOTS_DIR

# ─────────────────────────────────────────────
# TASK 1 — Load & explore
# ─────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

# The CSV header is "source_reference" (one combined column); rename to match spec
if "source_reference" in df.columns:
    df.rename(columns={"source_reference": "reference"}, inplace=True)

print("=" * 60)
print("TASK 1 — Dataset overview")
print("=" * 60)
print(f"\nShape: {df.shape}")
print(f"\nDtypes:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nkinetics value_counts:\n{df['kinetics'].value_counts()}")
print(f"\nn  — min: {df['n'].min()}, max: {df['n'].max()}")
print(f"k  — min: {df['k'].min()}, max: {df['k'].max()}")

# Histogram of n
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df['n'], bins=40, edgecolor='black', color='steelblue')
ax.set_xlabel('n')
ax.set_ylabel('Count')
ax.set_title('Histogram of n (kinetic exponent)')
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/hist_n.png", dpi=150)
plt.close(fig)
print("\nSaved: hist_n.png")

# Histogram of k
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df['k'], bins=50, edgecolor='black', color='darkorange')
ax.set_xlabel('k')
ax.set_ylabel('Count')
ax.set_title('Histogram of k (rate constant)')
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/hist_k.png", dpi=150)
plt.close(fig)
print("Saved: hist_k.png")

# Histogram of log10(k)
log10_k_vals = np.log10(df['k'])
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(log10_k_vals, bins=50, edgecolor='black', color='seagreen')
ax.set_xlabel('log₁₀(k)')
ax.set_ylabel('Count')
ax.set_title('Histogram of log₁₀(k)')
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/hist_log10_k.png", dpi=150)
plt.close(fig)
print("Saved: hist_log10_k.png")

# ─────────────────────────────────────────────
# TASK 2 — Engineered features
# ─────────────────────────────────────────────
df['protective_sum']     = df['Al'] + df['Cr'] + df['Si']
df['risk_sum']           = df['Zr'] + df['V']
df['Al_Cr_interaction']  = df['Al'] * df['Cr']
df['thermal_dose']       = df['To'] * df['t']

print("\n" + "=" * 60)
print("TASK 2 — Engineered features added")
print("=" * 60)
print(df[['protective_sum', 'risk_sum', 'Al_Cr_interaction', 'thermal_dose']].describe())

# ─────────────────────────────────────────────
# TASK 3 — Stage 1 target: kinetics_class
# ─────────────────────────────────────────────
def classify_n(n):
    if n <= 1.3:
        return 0   # Linear
    elif n <= 2.0:
        return 1   # Parabolic
    elif n <= 3.0:
        return 2   # Cubic
    else:
        return 3   # Quartic

df['kinetics_class'] = df['n'].apply(classify_n)

print("\n" + "=" * 60)
print("TASK 3 — kinetics_class distribution")
print("=" * 60)
label_map = {0: 'Linear', 1: 'Parabolic', 2: 'Cubic', 3: 'Quartic'}
dist = df['kinetics_class'].value_counts().sort_index()
for cls, cnt in dist.items():
    print(f"  {cls} ({label_map[cls]:>10s}): {cnt:3d}  ({cnt/len(df)*100:.1f}%)")

# ─────────────────────────────────────────────
# TASK 4 — Stage 2 target: log10_k
# ─────────────────────────────────────────────
df['log10_k'] = np.log10(df['k'])

print("\n" + "=" * 60)
print("TASK 4 — log10_k column added")
print("=" * 60)
print(df[['k', 'log10_k']].describe())

# ─────────────────────────────────────────────
# Final summary & save
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Final dataframe")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\nFirst 3 rows:\n{df.head(3).to_string()}")

out_csv = DATA_DIR / "processed_data.csv"
df.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")
