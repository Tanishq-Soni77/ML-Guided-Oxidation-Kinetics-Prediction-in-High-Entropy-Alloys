import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = _ROOT / "results" / "metrics"
PLOTS_DIR   = _ROOT / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES  = {0: 'Linear', 1: 'Parabolic', 2: 'HigherOrder'}
CLASS_COLORS = {0: 'steelblue', 1: 'darkorange', 2: 'seagreen'}

# ── Load predictions ──────────────────────────────────────────
df = pd.read_csv(f"{METRICS_DIR}/final_predictions.csv")

print(f"Loaded final_predictions.csv: {df.shape}")
print(df.head(3))

# ─────────────────────────────────────────────────────────────
# Plot 3 — Parity plot (raw k scale)
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))

for cls, name in CLASS_NAMES.items():
    mask = df['kinetics_class'] == cls
    ax.scatter(
        df.loc[mask, 'actual_k'],
        df.loc[mask, 'predicted_k'],
        label=name,
        color=CLASS_COLORS[cls],
        alpha=0.75,
        edgecolors='white',
        linewidths=0.4,
        s=60,
        zorder=3,
    )

# y = x reference line
all_k  = pd.concat([df['actual_k'], df['predicted_k']])
k_min  = all_k.min()
k_max  = all_k.max()
ax.plot([k_min, k_max], [k_min, k_max], 'k--', linewidth=1.2,
        label='y = x', zorder=2)

# R² annotation
ax.text(0.04, 0.93, r'$R^2 = 0.4260$', transform=ax.transAxes,
        fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Label 3 worst-predicted alloys
df['abs_err_k'] = (df['actual_k'] - df['predicted_k']).abs()
worst3 = df.nlargest(3, 'abs_err_k')

for _, row in worst3.iterrows():
    ax.annotate(
        row['alloy_group'],
        xy=(row['actual_k'], row['predicted_k']),
        xytext=(10, 10),
        textcoords='offset points',
        fontsize=7.5,
        arrowprops=dict(arrowstyle='->', color='dimgrey', lw=0.8),
        color='dimgrey',
    )

ax.set_xlabel(r"Actual $k$ (mg/cm²/s$^n$)", fontsize=12)
ax.set_ylabel(r"Predicted $k$ (mg/cm²/s$^n$)", fontsize=12)
ax.set_title("Stage 2 Parity Plot — XGBoost Regressor",
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.8)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/plot3_parity.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plot3_parity.png")

# ─────────────────────────────────────────────────────────────
# Plot 4 — Residual plot (log10 scale)
# ─────────────────────────────────────────────────────────────
df['residual'] = df['actual_log10k'] - df['predicted_log10k']

fig, ax = plt.subplots(figsize=(7, 5))

for cls, name in CLASS_NAMES.items():
    mask = df['kinetics_class'] == cls
    ax.scatter(
        df.loc[mask, 'predicted_log10k'],
        df.loc[mask, 'residual'],
        label=name,
        color=CLASS_COLORS[cls],
        alpha=0.75,
        edgecolors='white',
        linewidths=0.4,
        s=60,
        zorder=3,
    )

ax.axhline(0, color='black', linestyle='--', linewidth=1.2, zorder=2)

ax.set_xlabel(r"Predicted $\log_{10}(k)$", fontsize=12)
ax.set_ylabel(r"Residual (Actual $-$ Predicted)", fontsize=12)
ax.set_title(r"Residual Plot — $\log_{10}(k)$: Actual $-$ Predicted",
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.8)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/plot4_residuals.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plot4_residuals.png")
