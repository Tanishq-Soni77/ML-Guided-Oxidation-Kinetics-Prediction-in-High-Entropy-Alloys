import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
PLOTS_DIR = Path(__file__).resolve().parents[2] / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Plot 6 — Leakage comparison grouped bar chart
# ─────────────────────────────────────────────────────────────
thesis_log   = 0.9747
thesis_raw   = 0.9258
ours_log     = 0.7854
ours_raw     = 0.4260

groups       = [r"Stage 2 $R^2$ log$_{10}$(k)", r"Stage 2 $R^2$ raw $k$"]
thesis_vals  = [thesis_log, thesis_raw]
ours_vals    = [ours_log,   ours_raw]

x         = np.arange(len(groups))
bar_width  = 0.32

fig, ax = plt.subplots(figsize=(8, 5))

bars_thesis = ax.bar(x - bar_width / 2, thesis_vals, width=bar_width,
                     color='tomato',      alpha=0.88, label='Thesis: Random Split (leakage)',
                     edgecolor='white')
bars_ours   = ax.bar(x + bar_width / 2, ours_vals,   width=bar_width,
                     color='mediumseagreen', alpha=0.88, label='Ours: GroupShuffleSplit (honest)',
                     edgecolor='white')

# Value labels on top of each bar
for bar in bars_thesis:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
            f"{bar.get_height():.4f}", ha='center', va='bottom',
            fontsize=10, fontweight='bold', color='tomato')
for bar in bars_ours:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
            f"{bar.get_height():.4f}", ha='center', va='bottom',
            fontsize=10, fontweight='bold', color='seagreen')

# Perfect-score reference line
ax.axhline(1.0, color='black', linestyle='--', linewidth=1.0, alpha=0.6, label='Perfect score (R²=1.0)')

ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=12)
ax.set_ylabel(r"$R^2$ Score", fontsize=12)
ax.set_ylim(0, 1.12)
ax.set_title("Leakage Quantification: Random Split vs GroupShuffleSplit",
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.85)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/plot6_leakage_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plot6_leakage_comparison.png")

# ─────────────────────────────────────────────────────────────
# Final clean summary
# ─────────────────────────────────────────────────────────────
summary = """
============================================================
FINAL RESULTS SUMMARY
============================================================
Dataset: 297 real points | GroupShuffleSplit by alloy system | Zero composition leakage
Oversampling: class_weight='balanced' applied — no synthetic samples generated

STAGE 1 — Kinetics Classification (3-class: Linear / Parabolic / HigherOrder)
Best model     : ExtraTreesClassifier
Best params    : n_estimators=600, min_samples_split=5, max_depth=None, class_weight='balanced'
OOF macro-F1   : 0.9235
Test accuracy  : 0.7619
Test macro-F1  : 0.7531
Per-class F1   : Linear=0.4444  Parabolic=0.8148  HigherOrder=1.0000

STAGE 2 — Rate Constant Prediction
Best model     : XGBoost Regressor
Best params    : n_estimators=500, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8
CV R² log10(k) : -0.5030  (negative due to alloy-group distribution shift across folds)
Test R² log10(k): 0.7854
Test R² raw k  : 0.4260
Test RMSE raw k: 16.6097 mg/cm²
Test MAE raw k : 10.9196 mg/cm²

LEAKAGE QUANTIFICATION
Thesis random-split R² raw k : 0.9258
Our GroupSplit R² raw k      : 0.4260
Leakage inflation            : +0.4998 R² units (54% of reported R² is leakage)

Top 5 SHAP features (Stage 2):
  1. To            (Mean |SHAP| = 0.2248) — oxidation temperature
  2. predicted_n   (Mean |SHAP| = 0.1806) — kinetic exponent from Stage 1
  3. risk_sum      (Mean |SHAP| = 0.1349) — non-protective element sum
  4. Mo            (Mean |SHAP| = 0.0566) — volatile MoO3 former
  5. Cr            (Mean |SHAP| = 0.0516) — Cr2O3 protective scale former
============================================================
"""
print(summary)
