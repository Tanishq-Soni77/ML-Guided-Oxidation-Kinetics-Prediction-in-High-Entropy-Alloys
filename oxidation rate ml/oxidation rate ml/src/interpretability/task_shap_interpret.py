import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import shap

from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

from pathlib import Path
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
TARGET = 'log10_k'
SEED   = 42

FEATURE_COLS = [
    'Al', 'Si', 'Ti', 'V', 'Cr', 'Zr', 'Nb', 'Mo', 'Hf', 'Ta', 'W',
    'Tm', 'delta', 'To', 't',
    'protective_sum', 'risk_sum', 'Al_Cr_interaction', 'thermal_dose',
    'predicted_n',
]

INTERPRETATIONS = {
    'To'             : "Oxidation temperature — higher T drives faster oxide growth via Arrhenius kinetics.",
    'predicted_n'    : "Predicted kinetic exponent — encodes mechanism (linear/parabolic/higher-order) and directly scales k.",
    'Al'             : "Al content — promotes protective Al2O3 scale, suppressing oxidation rate at sufficient concentrations.",
    'protective_sum' : "Sum of protective-oxide-forming elements (Al+Cr+Si) — higher values correlate with slower oxidation.",
    'risk_sum'       : "Sum of non-protective elements (Ti+V+Nb+Mo+W) — higher values correlate with faster, non-protective oxidation.",
    't'              : "Exposure time — longer oxidation times allow scale thickening, modulating instantaneous rate constant.",
    'Cr'             : "Cr content — forms Cr2O3 protective scale; synergistic with Al for oxidation resistance.",
    'thermal_dose'   : "Product of temperature and time (T×t) — integrated thermal exposure driving cumulative oxide growth.",
    'Al_Cr_interaction': "Al×Cr cross term — captures synergistic protective scale formation between Al and Cr.",
    'Tm'             : "Alloy melting point — proxy for bond strength and diffusion barrier, influencing oxide growth rate.",
    'Mo'             : "Mo content — forms volatile MoO3 at high T, accelerating mass loss and increasing k.",
    'W'              : "W content — similar to Mo, prone to volatile oxide formation, elevating oxidation rate.",
    'V'              : "V content — V2O5 is a low-melting fluxing oxide that disrupts protective scale integrity.",
    'Nb'             : "Nb content — Nb2O5 growth is non-protective and contributes to higher oxidation rates.",
    'Ti'             : "Ti content — TiO2 is non-protective at high temperatures, increasing overall oxidation rate.",
    'delta'          : "Lattice distortion parameter — higher atomic size mismatch can affect diffusion through the oxide scale.",
    'Zr'             : "Zr content — ZrO2 can improve scale adhesion at low concentrations (reactive element effect).",
    'Hf'             : "Hf content — reactive element; small additions improve scale adhesion, reducing breakaway oxidation.",
    'Ta'             : "Ta content — forms Ta2O5, generally non-protective but less volatile than Mo/W oxides.",
    'Si'             : "Si content — SiO2 sub-scale enhances oxidation resistance at intermediate concentrations.",
}

# ── Load data ─────────────────────────────────────────────────
train = pd.read_csv(f"{DATA_DIR}/train_data_s2.csv")
test  = pd.read_csv(f"{DATA_DIR}/test_data_s2.csv")

train = train.rename(columns={'predicted_n_train': 'predicted_n'})
test  = test.rename(columns={'predicted_n_test':  'predicted_n'})

X_train_raw = train[FEATURE_COLS].values
y_train     = train[TARGET].values
X_test_raw  = test[FEATURE_COLS].values

print(f"Train: {X_train_raw.shape}  |  Test: {X_test_raw.shape}")

# ─────────────────────────────────────────────────────────────
# Step 1 — RobustScaler + retrain XGBoost on full train
# ─────────────────────────────────────────────────────────────
scaler = RobustScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_raw), columns=FEATURE_COLS
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_raw), columns=FEATURE_COLS
)

model = XGBRegressor(
    n_estimators    = 500,
    max_depth       = 3,
    learning_rate   = 0.1,
    subsample       = 0.8,
    colsample_bytree= 0.8,
    reg_alpha       = 0,
    reg_lambda      = 1,
    min_child_weight= 3,
    objective       = 'reg:squarederror',
    eval_metric     = 'rmse',
    verbosity       = 0,
    random_state    = SEED,
    n_jobs          = 1,
)
model.fit(X_train_scaled, y_train)
print("Model trained on full train set.")

# ─────────────────────────────────────────────────────────────
# Step 2 — SHAP TreeExplainer + rank by mean |SHAP|
# ─────────────────────────────────────────────────────────────
booster     = model.get_booster()
explainer   = shap.TreeExplainer(booster)
shap_values = explainer.shap_values(X_test_scaled)

mean_abs_shap = (
    pd.DataFrame(np.abs(shap_values), columns=FEATURE_COLS)
    .mean()
    .sort_values(ascending=False)
)

# ─────────────────────────────────────────────────────────────
# Step 3 — Print ranked table of ALL features
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("All features ranked by mean absolute SHAP value")
print("=" * 50)
print(f"{'Rank':<5}  {'Feature':<22}  {'Mean |SHAP|':>11}")
print("-" * 42)
for rank, (feature, value) in enumerate(mean_abs_shap.items(), start=1):
    print(f"{rank:<5}  {feature:<22}  {value:>11.4f}")

# ─────────────────────────────────────────────────────────────
# Step 4 — Top 5 with physical interpretation
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("Top 5 features — physical interpretation")
print("=" * 75)
for rank, (feature, value) in enumerate(mean_abs_shap.head(5).items(), start=1):
    interp = INTERPRETATIONS.get(feature, "No interpretation defined for this feature.")
    print(f"\nRank {rank}")
    print(f"Feature: {feature} | Mean |SHAP|: {value:.4f} | Interpretation: {interp}")
