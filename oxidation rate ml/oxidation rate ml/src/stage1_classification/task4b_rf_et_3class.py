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
CLASS_NAMES = ['Linear', 'Parabolic', 'HigherOrder']
SEED        = 42

# ── Load data ────────────────────────────────────────────────
train = pd.read_csv(f"{DATA_DIR}/train_scaled_3class.csv")
test  = pd.read_csv(f"{DATA_DIR}/test_scaled_3class.csv")

feature_cols = [c for c in train.columns if c != TARGET]

X_train = train[feature_cols].values
y_train = train[TARGET].values
X_test  = test[feature_cols].values
y_test  = test[TARGET].values

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

# ── Shared param grid ─────────────────────────────────────────
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
        print(f"  {cls:>11s}: {report[cls]['f1-score']:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title(f"{label} 3-class — Confusion Matrix", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{cm_fname}", dpi=150)
    plt.close()
    print(f"\nConfusion matrix saved: {cm_fname}")

    # Save model
    with open(f"{MODELS_DIR}/{pkl_fname}", 'wb') as f:
        pickle.dump(best, f)
    print(f"Model saved          : {pkl_fname}")

    return {
        'label'   : label,
        'cv_f1'   : search.best_score_,
        'test_acc': acc,
        'test_mf1': mf1,
        'ho_f1'   : report['HigherOrder']['f1-score'],
    }

# ── Model 2: Random Forest ────────────────────────────────────
res_rf = run_search_and_report(
    RandomForestClassifier(random_state=SEED),
    label    = "Random Forest 3-class",
    cm_fname = "cm_rf_3class.png",
    pkl_fname= "model_rf_3class.pkl",
)

# ── Model 3: ExtraTrees ───────────────────────────────────────
res_et = run_search_and_report(
    ExtraTreesClassifier(random_state=SEED),
    label    = "ExtraTrees 3-class",
    cm_fname = "cm_et_3class.png",
    pkl_fname= "model_et_3class.pkl",
)

# ── Final summary (all 3 models including XGBoost reference) ──
print("\n" + "=" * 65)
print("FINAL SUMMARY — 3-class models (RF + ET; XGBoost in task3b)")
print("=" * 65)
print(f"{'Model':<22}  {'CV F1':>6}  {'Test Acc':>8}  {'Test F1':>7}  {'HigherOrder F1':>14}")
print("-" * 65)
for r in [res_rf, res_et]:
    print(f"{r['label']:<22}  {r['cv_f1']:>6.4f}  {r['test_acc']:>8.4f}  "
          f"{r['test_mf1']:>7.4f}  {r['ho_f1']:>14.4f}")
print("\nNote: compare with model_xgb_3class results from task3b_xgb_3class.py")
