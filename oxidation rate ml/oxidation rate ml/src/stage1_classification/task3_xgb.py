import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR   = _ROOT / "data"
PLOTS_DIR  = _ROOT / "results" / "plots"
MODELS_DIR = _ROOT / "models"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TARGET       = 'kinetics_class'
CLASS_NAMES  = ['Linear', 'Parabolic', 'Cubic', 'Quartic']
SEED         = 42

# ── Load data ────────────────────────────────────────────────
train = pd.read_csv(f"{DATA_DIR}/train_scaled.csv")
test  = pd.read_csv(f"{DATA_DIR}/test_scaled.csv")

feature_cols = [c for c in train.columns if c != TARGET]

X_train = train[feature_cols].values
y_train = train[TARGET].values
X_test  = test[feature_cols].values
y_test  = test[TARGET].values

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

# ── Sample weights (Strategy D: class_weight='balanced') ─────
sample_weights = compute_sample_weight('balanced', y_train)

# ── Param grid ───────────────────────────────────────────────
param_grid = {
    'n_estimators'    : [200, 300, 400, 500],
    'max_depth'       : [3, 4, 5, 6],
    'learning_rate'   : [0.01, 0.05, 0.1, 0.15],
    'subsample'       : [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
}

base = XGBClassifier(
    eval_metric='mlogloss',
    verbosity=0,
    random_state=SEED,
    n_jobs=1,
)

search = RandomizedSearchCV(
    base, param_grid,
    n_iter=15, cv=3, scoring='f1_macro',
    random_state=SEED, n_jobs=1, refit=True,
    verbose=1,
)
search.fit(X_train, y_train, sample_weight=sample_weights)

# ── Results ──────────────────────────────────────────────────
best   = search.best_estimator_
y_pred = best.predict(X_test)

acc  = accuracy_score(y_test, y_pred)
mf1  = f1_score(y_test, y_pred, average='macro', zero_division=0)

print("\n" + "=" * 55)
print("XGBoost — RandomizedSearchCV Results")
print("=" * 55)
print(f"Best hyperparameters : {search.best_params_}")
print(f"CV macro-F1 (train)  : {search.best_score_:.4f}")
print(f"Test accuracy        : {acc:.4f}")
print(f"Test macro-F1        : {mf1:.4f}")
print("\nPer-class F1:")
report = classification_report(
    y_test, y_pred, target_names=CLASS_NAMES,
    zero_division=0, output_dict=True
)
for cls in CLASS_NAMES:
    print(f"  {cls:>10s}: {report[cls]['f1-score']:.4f}")

# ── Confusion matrix ─────────────────────────────────────────
cm   = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(ax=ax, colorbar=True, cmap='Blues')
ax.set_title("XGBoost — Confusion Matrix", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/cm_xgb.png", dpi=150)
plt.close()
print(f"\nConfusion matrix saved: cm_xgb.png")

# ── Save model ───────────────────────────────────────────────
with open(f"{MODELS_DIR}/model_xgb.pkl", 'wb') as f:
    pickle.dump(best, f)
print(f"Model saved          : model_xgb.pkl")
