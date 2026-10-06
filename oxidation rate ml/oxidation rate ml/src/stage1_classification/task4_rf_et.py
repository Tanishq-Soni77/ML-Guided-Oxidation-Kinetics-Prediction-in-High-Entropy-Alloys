import pickle
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR   = _ROOT / "data"
PLOTS_DIR  = _ROOT / "results" / "plots"
MODELS_DIR = _ROOT / "models"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TARGET      = 'kinetics_class'
CLASS_NAMES = ['Linear', 'Parabolic', 'Cubic', 'Quartic']
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

# ── Shared param grid structure ───────────────────────────────
tree_param_grid = {
    'n_estimators'    : [200, 400, 600],
    'max_depth'       : [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'class_weight'    : ['balanced', None],
}

# ── Helper ────────────────────────────────────────────────────
def run_search_and_report(estimator, label, cm_fname, pkl_fname):
    search = RandomizedSearchCV(
        estimator, tree_param_grid,
        n_iter=15, cv=3, scoring='f1_macro',
        random_state=SEED, n_jobs=1, refit=True,
        verbose=1,
    )
    search.fit(X_train, y_train)

    best   = search.best_estimator_
    y_pred = best.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    mf1    = f1_score(y_test, y_pred, average='macro', zero_division=0)

    print("\n" + "=" * 55)
    print(f"{label} — RandomizedSearchCV Results")
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

    # Confusion matrix
    cm   = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title(f"{label} — Confusion Matrix", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{cm_fname}", dpi=150)
    plt.close()
    print(f"\nConfusion matrix saved: {cm_fname}")

    # Save model
    with open(f"{MODELS_DIR}/{pkl_fname}", 'wb') as f:
        pickle.dump(best, f)
    print(f"Model saved          : {pkl_fname}")

    return best, acc, mf1

# ── Model 2: Random Forest ────────────────────────────────────
rf_model, rf_acc, rf_mf1 = run_search_and_report(
    RandomForestClassifier(random_state=SEED),
    label    = "Random Forest",
    cm_fname = "cm_rf.png",
    pkl_fname= "model_rf.pkl",
)

# ── Model 3: ExtraTrees ───────────────────────────────────────
et_model, et_acc, et_mf1 = run_search_and_report(
    ExtraTreesClassifier(random_state=SEED),
    label    = "ExtraTrees",
    cm_fname = "cm_et.png",
    pkl_fname= "model_et.pkl",
)

# ── Side-by-side summary ─────────────────────────────────────
print("\n" + "=" * 55)
print("SUMMARY")
print("=" * 55)
print(f"{'Model':<16}  {'Test Acc':>10}  {'Test macro-F1':>13}")
print("-" * 44)
print(f"{'Random Forest':<16}  {rf_acc:>10.4f}  {rf_mf1:>13.4f}")
print(f"{'ExtraTrees':<16}  {et_acc:>10.4f}  {et_mf1:>13.4f}")
