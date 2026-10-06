"""
hea_fixes.py
============
Four targeted fixes for the HEA oxidation-kinetics ML pipeline.

Run:  python hea_fixes.py

Outputs:  fix1.txt  fix2.txt  fix2.png  fix3.txt  fix4.txt
"""

# ─────────────────────────────────────────────────────────────────
# 0. Dependency check
# ─────────────────────────────────────────────────────────────────
import sys, os

REQUIRED = {
    'numpy'     : 'numpy',
    'pandas'    : 'pandas',
    'sklearn'   : 'scikit-learn',
    'xgboost'   : 'xgboost',
    'matplotlib': 'matplotlib',
}
missing = [pkg for imp, pkg in REQUIRED.items()
           if not __import__('importlib').util.find_spec(imp)]
if missing:
    print(f"[ERROR] Missing packages. Run:\n  pip install {' '.join(missing)}")
    sys.exit(1)

import warnings
warnings.filterwarnings('ignore')

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection      import (GroupShuffleSplit, GroupKFold,
                                           RandomizedSearchCV, train_test_split)
from sklearn.ensemble             import (ExtraTreesClassifier,
                                           ExtraTreesRegressor,
                                           RandomForestRegressor)
from sklearn.utils.class_weight   import compute_sample_weight
from sklearn.metrics              import (accuracy_score, f1_score,
                                           r2_score, mean_squared_error,
                                           mean_absolute_error)
from xgboost import XGBRegressor

# ═════════════════════════════════════════════════════════════════
# GLOBAL SETUP
# ═════════════════════════════════════════════════════════════════
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR    = _ROOT / "data"
METRICS_DIR = _ROOT / "results" / "metrics"
PLOTS_DIR   = _ROOT / "results" / "plots"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
SEED       = 42
CLASS_NAMES = ['Linear', 'Parabolic', 'HigherOrder']
N_MAP       = {0: 1.0, 1: 2.0, 2: 3.0}      # n_class int → float for S2

# Stated best params (spec defaults; Fix 1 may improve them)
SPEC_S1 = dict(n_estimators=600, max_depth=None, min_samples_split=5,
               class_weight='balanced', random_state=SEED, n_jobs=-1)
SPEC_S2 = dict(n_estimators=500, max_depth=3, learning_rate=0.1,
               subsample=0.8, colsample_bytree=0.8,
               objective='reg:squarederror', eval_metric='rmse',
               verbosity=0, random_state=SEED, n_jobs=1)

# Feature lists
S1_FEATS = ['Al','Si','Ti','V','Cr','Zr','Nb','Mo','Hf','Ta','W',
            'Tm','delta','To','t',
            'protective_sum','risk_sum','thermal_dose']       # 18 features

FEATS_15  = ['Ti','V','Cr','Zr','Nb','Mo','Hf','Ta','W',
             'Al','Si','Tm','delta','To','t']                 # Fix 4 baseline

# Param search spaces (identical to original scripts, Al_Cr_interaction excluded)
ET_GRID = {
    'n_estimators'    : [200, 400, 600],
    'max_depth'       : [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'class_weight'    : ['balanced', None],
}
XGB_GRID = {
    'n_estimators'    : [300, 500, 700],
    'max_depth'       : [3, 4, 5],
    'learning_rate'   : [0.01, 0.03, 0.05, 0.1],
    'subsample'       : [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'reg_alpha'       : [0, 0.1, 0.5, 1.0],
    'reg_lambda'      : [1, 2, 5],
    'min_child_weight': [1, 3, 5],
}

SEP  = "═" * 70
SEP2 = "─" * 70

# ─────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, 'processed_data.csv'))
    # Merge alloy_group from pre-split files (row-aligned with processed_data)
    tr  = pd.read_csv(os.path.join(DATA_DIR, 'train_data.csv'))
    te  = pd.read_csv(os.path.join(DATA_DIR, 'test_data.csv'))
    ref = pd.concat([tr, te], ignore_index=True)
    key = ['Al','Si','Ti','V','Cr','Zr','Nb','Mo','Hf','Ta','W',
           'Tm','delta','To','t','n','k','kinetics_class','log10_k']
    df  = df.merge(ref[key + ['alloy_group']].drop_duplicates(subset=key),
                   on=key, how='left')
    assert df['alloy_group'].notna().all(), "alloy_group merge failed"
    # 3-class target: merge Cubic(2) + Quartic(3) → HigherOrder(2)
    df['n_class'] = df['kinetics_class'].map({0: 0, 1: 1, 2: 2, 3: 2})
    return df

df_all = load_data()
print(f"\nDataset loaded: {df_all.shape[0]} rows | "
      f"{df_all['alloy_group'].nunique()} alloy groups")
print(f"n_class: { {i: CLASS_NAMES[i] for i in range(3)} }")
print(f"Distribution: { df_all['n_class'].value_counts().sort_index().to_dict() }")

# Canonical seed-42 split (used by Fix 1, 3)
def make_split(seed=SEED):
    g   = df_all['alloy_group'].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    ti, ei = next(gss.split(df_all, groups=g))
    return ti, ei

TR42, TE42 = make_split(SEED)

# ─────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────
def s1_data(idx, extra_feats=None):
    feats = S1_FEATS + (extra_feats or [])
    return (df_all.iloc[idx][feats].values,
            df_all.iloc[idx]['n_class'].values)

def s2_data(idx, feats):
    return (df_all.iloc[idx][feats].values,
            df_all.iloc[idx]['log10_k'].values)

def eval_s1(yt, yp):
    acc = accuracy_score(yt, yp)
    mf1 = f1_score(yt, yp, average='macro', zero_division=0)
    pf1 = f1_score(yt, yp, average=None,   zero_division=0, labels=[0, 1, 2])
    return acc, mf1, dict(zip(CLASS_NAMES, pf1))

def eval_s2(yt_log, yp_log):
    r2l  = r2_score(yt_log, yp_log)
    yr   = 10 ** yt_log;  ypr = 10 ** yp_log
    r2r  = r2_score(yr, ypr)
    rmse = float(np.sqrt(mean_squared_error(yr, ypr)))
    mae  = float(mean_absolute_error(yr, ypr))
    return r2l, r2r, rmse, mae

def oof_predicted_n(X_tr, y_tr, g_tr, s1_params, seed=SEED):
    """GroupKFold OOF predicted_n on training set (for leak-free S2 train)."""
    oof  = np.full(len(y_tr), -1, dtype=int)
    gkf  = GroupKFold(n_splits=5)
    for _, (fi, vi) in enumerate(gkf.split(X_tr, groups=g_tr)):
        sw = compute_sample_weight('balanced', y_tr[fi])
        m  = ExtraTreesClassifier(**{**s1_params, 'random_state': seed})
        m.fit(X_tr[fi], y_tr[fi], sample_weight=sw)
        oof[vi] = m.predict(X_tr[vi])
    return np.array([N_MAP[c] for c in oof])

def write_txt(fname, lines):
    path = os.path.join(METRICS_DIR, fname)
    with open(path, 'w') as f:
        f.write('\n'.join(str(l) for l in lines) + '\n')
    print(f"  → Saved: {fname}")

# Running best-param store (initialised to spec values, updated by Fix 1)
best_s1_params = SPEC_S1.copy()
best_s2_params = SPEC_S2.copy()

# ═════════════════════════════════════════════════════════════════
# FIX 1 — GroupKFold hyperparameter tuning
# ═════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("FIX 1 — GroupKFold(5) Hyperparameter Tuning")
print(f"{SEP}\n")

try:
    X1_tr, y1_tr = s1_data(TR42)
    X1_te, y1_te = s1_data(TE42)
    g_tr = df_all.iloc[TR42]['alloy_group'].values
    sw1  = compute_sample_weight('balanced', y1_tr)

    # ── Stage 1: ExtraTreesClassifier ────────────────────────────
    print("[Stage 1] ExtraTreesClassifier | GroupKFold(5) | f1_macro")
    s1_search = RandomizedSearchCV(
        ExtraTreesClassifier(random_state=SEED, n_jobs=-1),
        ET_GRID,
        n_iter=30, cv=GroupKFold(n_splits=5),
        scoring='f1_macro',
        random_state=SEED, n_jobs=-1, refit=True,
    )
    s1_search.fit(X1_tr, y1_tr, sample_weight=sw1, groups=g_tr)
    best_s1_params = {**s1_search.best_params_,
                      'random_state': SEED, 'n_jobs': -1}

    s1_pred_te = s1_search.best_estimator_.predict(X1_te)
    f1_acc, f1_mf1, f1_pc = eval_s1(y1_te, s1_pred_te)
    print(f"  Best params  : {s1_search.best_params_}")
    print(f"  CV macro-F1  : {s1_search.best_score_:.4f}")
    print(f"  Test accuracy: {f1_acc:.4f}  |  Test macro-F1: {f1_mf1:.4f}")
    for cls, v in f1_pc.items():
        print(f"    {cls:<14}: {v:.4f}")

    # ── Build leak-free S2 training set using OOF predicted_n ────
    pn_tr_oof = oof_predicted_n(X1_tr, y1_tr, g_tr, best_s1_params)
    pn_te     = np.array([N_MAP[c] for c in s1_pred_te])
    X2_tr = np.c_[X1_tr, pn_tr_oof]
    X2_te = np.c_[X1_te, pn_te]
    y2_tr = df_all.iloc[TR42]['log10_k'].values
    y2_te = df_all.iloc[TE42]['log10_k'].values

    # ── Stage 2: XGBoost Regressor ───────────────────────────────
    print("\n[Stage 2] XGBoost Regressor    | GroupKFold(5) | r2")
    s2_search = RandomizedSearchCV(
        XGBRegressor(objective='reg:squarederror', eval_metric='rmse',
                     verbosity=0, random_state=SEED, n_jobs=1),
        XGB_GRID,
        n_iter=20, cv=GroupKFold(n_splits=5),
        scoring='r2',
        random_state=SEED, n_jobs=1, refit=True,
    )
    s2_search.fit(X2_tr, y2_tr, groups=g_tr)
    best_s2_params = {**s2_search.best_params_,
                      'objective': 'reg:squarederror', 'eval_metric': 'rmse',
                      'verbosity': 0, 'random_state': SEED, 'n_jobs': 1}

    yp2 = s2_search.best_estimator_.predict(X2_te)
    r2l, r2r, rmse, mae = eval_s2(y2_te, yp2)
    print(f"  Best params    : {s2_search.best_params_}")
    print(f"  CV R²          : {s2_search.best_score_:.4f}")
    print(f"  Test R²(log-k) : {r2l:.4f}")
    print(f"  Test R²(raw-k) : {r2r:.4f}")
    print(f"  Test RMSE      : {rmse:.6f}")
    print(f"  Test MAE       : {mae:.6f}")

    lines = [
        "FIX 1 — GroupKFold(5) Hyperparameter Tuning",
        "=" * 55, "",
        "[Stage 1] ExtraTreesClassifier | GroupKFold(5) | f1_macro",
        f"  Best params  : {s1_search.best_params_}",
        f"  CV macro-F1  : {s1_search.best_score_:.4f}",
        f"  Test accuracy: {f1_acc:.4f}  |  Test macro-F1: {f1_mf1:.4f}",
    ] + [f"    {c:<14}: {v:.4f}" for c, v in f1_pc.items()] + [
        "",
        "[Stage 2] XGBoost Regressor | GroupKFold(5) | r2",
        f"  Best params    : {s2_search.best_params_}",
        f"  CV R²          : {s2_search.best_score_:.4f}",
        f"  Test R²(log-k) : {r2l:.4f}",
        f"  Test R²(raw-k) : {r2r:.4f}",
        f"  Test RMSE      : {rmse:.6f}",
        f"  Test MAE       : {mae:.6f}",
    ]
    write_txt('fix1.txt', lines)
    print("\n[Fix 1 DONE]")

except Exception as exc:
    print(f"[Fix 1 FAILED]: {exc}")
    import traceback; traceback.print_exc()

# ═════════════════════════════════════════════════════════════════
# FIX 2 — Multi-seed robustness (seeds 0–9)
# ═════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("FIX 2 — Multi-Seed Robustness (seeds 0–9, Fix1 params, no re-tuning)")
print(f"{SEP}\n")

try:
    records = []
    hdr = (f"{'Seed':>4}  {'S1 acc':>7}  {'S1 mF1':>7}  "
           f"{'R²(log-k)':>10}  {'R²(raw-k)':>10}  {'RMSE':>10}  {'MAE':>10}")
    print(hdr)
    print(SEP2)

    for seed in range(10):
        tri, tei = make_split(seed)
        g_tri    = df_all.iloc[tri]['alloy_group'].values

        # S1
        Xa_tr, ya_tr = s1_data(tri)
        Xa_te, ya_te = s1_data(tei)
        sw_s         = compute_sample_weight('balanced', ya_tr)

        s1_m = ExtraTreesClassifier(**{**best_s1_params, 'random_state': seed})
        s1_m.fit(Xa_tr, ya_tr, sample_weight=sw_s)
        yp_te_cls  = s1_m.predict(Xa_te)
        sacc, smf1, _ = eval_s1(ya_te, yp_te_cls)

        # OOF predicted_n for S2 train; direct predict for S2 test
        pn_tr_s = oof_predicted_n(Xa_tr, ya_tr, g_tri, best_s1_params, seed=seed)
        pn_te_s = np.array([N_MAP[c] for c in yp_te_cls])

        Xb_tr = np.c_[Xa_tr, pn_tr_s]
        Xb_te = np.c_[Xa_te, pn_te_s]
        yb_tr = df_all.iloc[tri]['log10_k'].values
        yb_te = df_all.iloc[tei]['log10_k'].values

        # S2
        s2_m = XGBRegressor(**{**best_s2_params, 'random_state': seed})
        s2_m.fit(Xb_tr, yb_tr)
        r2l_s, r2r_s, rmse_s, mae_s = eval_s2(yb_te, s2_m.predict(Xb_te))

        records.append(dict(seed=seed, s1_acc=sacc, s1_mf1=smf1,
                            r2_log=r2l_s, r2_raw=r2r_s,
                            rmse=rmse_s, mae=mae_s))
        print(f"{seed:>4}  {sacc:>7.4f}  {smf1:>7.4f}  "
              f"{r2l_s:>10.4f}  {r2r_s:>10.4f}  {rmse_s:>10.6f}  {mae_s:>10.6f}")

    # Summary stats
    df_rec = pd.DataFrame(records)
    mu  = df_rec.mean(numeric_only=True)
    std = df_rec.std(numeric_only=True)
    print(SEP2)
    print(f"{'Mean':>4}  {mu.s1_acc:>7.4f}  {mu.s1_mf1:>7.4f}  "
          f"{mu.r2_log:>10.4f}  {mu.r2_raw:>10.4f}  "
          f"{mu.rmse:>10.6f}  {mu.mae:>10.6f}")
    print(f"{'Std':>4}  {std.s1_acc:>7.4f}  {std.s1_mf1:>7.4f}  "
          f"{std.r2_log:>10.4f}  {std.r2_raw:>10.4f}  "
          f"{std.rmse:>10.6f}  {std.mae:>10.6f}")

    # ── Bar chart ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    seeds_x = df_rec['seed'].values
    r2_vals = df_rec['r2_log'].values
    colors  = ['#2196F3' if v >= 0 else '#F44336' for v in r2_vals]
    bars    = ax.bar(seeds_x, r2_vals, color=colors, edgecolor='white',
                     linewidth=0.8, zorder=3)
    ax.axhline(mu.r2_log, color='#FF6F00', linewidth=2,
               linestyle='--', label=f'Mean R² = {mu.r2_log:.4f}')
    ax.axhline(mu.r2_log + std.r2_log, color='#FF6F00', linewidth=1,
               linestyle=':', alpha=0.6)
    ax.axhline(mu.r2_log - std.r2_log, color='#FF6F00', linewidth=1,
               linestyle=':', alpha=0.6)
    for bar, v in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_xlabel('Random Seed', fontsize=11)
    ax.set_ylabel('Test R²  (log₁₀ k)', fontsize=11)
    ax.set_title('Fix 2 — Multi-Seed Robustness: S2 Test R² per Seed\n'
                 '(GroupShuffleSplit + Fix1 best params, no re-tuning)',
                 fontsize=11, fontweight='bold')
    ax.set_xticks(seeds_x)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.4, zorder=0)
    ax.set_ylim(bottom=min(0, r2_vals.min() - 0.05))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'fix2.png'), dpi=150)
    plt.close()
    print("  → Saved: fix2.png")

    # ── Text report ──────────────────────────────────────────────
    lines = [
        "FIX 2 — Multi-Seed Robustness (seeds 0–9, Fix1 best params)",
        "=" * 55, "",
        hdr,
        SEP2,
    ]
    for r in records:
        lines.append(f"{r['seed']:>4}  {r['s1_acc']:>7.4f}  {r['s1_mf1']:>7.4f}  "
                     f"{r['r2_log']:>10.4f}  {r['r2_raw']:>10.4f}  "
                     f"{r['rmse']:>10.6f}  {r['mae']:>10.6f}")
    lines += [
        SEP2,
        f"{'Mean':>4}  {mu.s1_acc:>7.4f}  {mu.s1_mf1:>7.4f}  "
        f"{mu.r2_log:>10.4f}  {mu.r2_raw:>10.4f}  "
        f"{mu.rmse:>10.6f}  {mu.mae:>10.6f}",
        f"{'Std':>4}  {std.s1_acc:>7.4f}  {std.s1_mf1:>7.4f}  "
        f"{std.r2_log:>10.4f}  {std.r2_raw:>10.4f}  "
        f"{std.rmse:>10.6f}  {std.mae:>10.6f}",
        "",
        f"S1 macro-F1 : {mu.s1_mf1:.4f} ± {std.s1_mf1:.4f}",
        f"S2 R²(log-k): {mu.r2_log:.4f} ± {std.r2_log:.4f}",
        f"S2 R²(raw-k): {mu.r2_raw:.4f} ± {std.r2_raw:.4f}",
    ]
    write_txt('fix2.txt', lines)
    print("\n[Fix 2 DONE]")

except Exception as exc:
    print(f"[Fix 2 FAILED]: {exc}")
    import traceback; traceback.print_exc()

# ═════════════════════════════════════════════════════════════════
# FIX 3 — Ablation study
# ═════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("FIX 3 — Ablation: S2 feature-set comparison (seed=42 split)")
print(f"{SEP}\n")

try:
    X1_tr, y1_tr = s1_data(TR42)
    X1_te, y1_te = s1_data(TE42)
    g_tr42 = df_all.iloc[TR42]['alloy_group'].values
    sw42   = compute_sample_weight('balanced', y1_tr)
    y2_tr  = df_all.iloc[TR42]['log10_k'].values
    y2_te  = df_all.iloc[TE42]['log10_k'].values

    # S1 fit once (same for A and C)
    s1_ab = ExtraTreesClassifier(**best_s1_params)
    s1_ab.fit(X1_tr, y1_tr, sample_weight=sw42)

    pn_tr_oof = oof_predicted_n(X1_tr, y1_tr, g_tr42, best_s1_params)
    pn_te_ab  = np.array([N_MAP[c] for c in s1_ab.predict(X1_te)])
    s1_acc_ab, s1_mf1_ab, _ = eval_s1(y1_te, s1_ab.predict(X1_te))

    # Oracle n: true n_class mapped to float (upper-bound for S2)
    oracle_tr = np.array([N_MAP[c] for c in df_all.iloc[TR42]['n_class'].values])
    oracle_te = np.array([N_MAP[c] for c in df_all.iloc[TE42]['n_class'].values])

    ablation_results = {}

    def run_s2(X_tr_feat, X_te_feat, label):
        m = XGBRegressor(**best_s2_params)
        m.fit(X_tr_feat, y2_tr)
        return eval_s2(y2_te, m.predict(X_te_feat))

    # A) 18 base features + predicted_n from S1  (current pipeline)
    Xa_tr = np.c_[X1_tr, pn_tr_oof]
    Xa_te = np.c_[X1_te, pn_te_ab]
    ablation_results['A: 18-feat + predicted_n (S1)'] = run_s2(Xa_tr, Xa_te, 'A')

    # B) 18 base features only, no S1
    ablation_results['B: 18-feat only  (no S1)      '] = run_s2(X1_tr, X1_te, 'B')

    # C) 18 base features + true n_class label (oracle upper bound)
    Xc_tr = np.c_[X1_tr, oracle_tr]
    Xc_te = np.c_[X1_te, oracle_te]
    ablation_results['C: 18-feat + true n_class (oracle)'] = run_s2(Xc_tr, Xc_te, 'C')

    hdr3 = (f"  {'Variant':<38}  {'R²(log-k)':>10}  {'R²(raw-k)':>10}"
            f"  {'RMSE':>10}  {'MAE':>10}")
    print(f"  S1 macro-F1 (used in A & C): {s1_mf1_ab:.4f}")
    print()
    print(hdr3)
    print("  " + SEP2)
    lines3 = [
        "FIX 3 — Ablation: S2 Feature-Set Comparison (seed=42)",
        "=" * 55, "",
        f"  S1 macro-F1 (used in A & C): {s1_mf1_ab:.4f}", "",
        hdr3, "  " + SEP2,
    ]
    for label, (r2l, r2r, rmse, mae) in ablation_results.items():
        row = (f"  {label:<38}  {r2l:>10.4f}  {r2r:>10.4f}"
               f"  {rmse:>10.6f}  {mae:>10.6f}")
        print(row)
        lines3.append(row)

    lines3 += [
        "", "─" * 55,
        "A = current pipeline (S1 → predicted_n → S2)",
        "B = S2 trained on composition only, no kinetics class input",
        "C = oracle upper bound (true label fed to S2, not predicted)",
        "Gap A→C = information ceiling from perfect S1 classification",
        "Gap B→A = marginal gain from adding S1 predicted_n",
    ]
    write_txt('fix3.txt', lines3)
    print("\n[Fix 3 DONE]")

except Exception as exc:
    print(f"[Fix 3 FAILED]: {exc}")
    import traceback; traceback.print_exc()

# ═════════════════════════════════════════════════════════════════
# FIX 4 — Leakage isolation
# ═════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("FIX 4 — Leakage Isolation (random vs group split, simple vs full pipeline)")
print(f"{SEP}\n")

try:
    y_all_log = df_all['log10_k'].values

    # ── Exp 1: Random row split, ExtraTreesRegressor & RandomForestRegressor ──
    # 15 raw features, no engineered features, no S1
    X15 = df_all[FEATS_15].values
    X15_tr_r, X15_te_r, y_tr_r, y_te_r = train_test_split(
        X15, y_all_log, test_size=0.2, random_state=SEED
    )
    etr_rand = ExtraTreesRegressor(n_estimators=500, random_state=SEED, n_jobs=-1)
    etr_rand.fit(X15_tr_r, y_tr_r)
    exp1_etr = eval_s2(y_te_r, etr_rand.predict(X15_te_r))

    rf_rand = RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1)
    rf_rand.fit(X15_tr_r, y_tr_r)
    exp1_rf = eval_s2(y_te_r, rf_rand.predict(X15_te_r))

    # ── Exp 2: GroupShuffleSplit, same ETR & RF, same 15 features ──────────
    X15_tr_g = df_all.iloc[TR42][FEATS_15].values
    X15_te_g = df_all.iloc[TE42][FEATS_15].values
    y_tr_g   = df_all.iloc[TR42]['log10_k'].values
    y_te_g   = df_all.iloc[TE42]['log10_k'].values

    etr_grp = ExtraTreesRegressor(n_estimators=500, random_state=SEED, n_jobs=-1)
    etr_grp.fit(X15_tr_g, y_tr_g)
    exp2_etr = eval_s2(y_te_g, etr_grp.predict(X15_te_g))

    rf_grp = RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1)
    rf_grp.fit(X15_tr_g, y_tr_g)
    exp2_rf = eval_s2(y_te_g, rf_grp.predict(X15_te_g))

    # ── Exp 3: GroupShuffleSplit, ETC→XGB, 19 features (full pipeline) ──────
    Xa_tr3, ya_tr3 = s1_data(TR42)
    Xa_te3, ya_te3 = s1_data(TE42)
    g_tr3          = df_all.iloc[TR42]['alloy_group'].values
    sw3            = compute_sample_weight('balanced', ya_tr3)
    y_te_log3      = df_all.iloc[TE42]['log10_k'].values

    s1_exp3 = ExtraTreesClassifier(**best_s1_params)
    s1_exp3.fit(Xa_tr3, ya_tr3, sample_weight=sw3)
    s1_pred3        = s1_exp3.predict(Xa_te3)
    s1_acc3, s1_mf1_3, _ = eval_s1(ya_te3, s1_pred3)

    pn_tr3 = oof_predicted_n(Xa_tr3, ya_tr3, g_tr3, best_s1_params)
    pn_te3 = np.array([N_MAP[c] for c in s1_pred3])

    s2_exp3 = XGBRegressor(**best_s2_params)
    s2_exp3.fit(np.c_[Xa_tr3, pn_tr3],
                df_all.iloc[TR42]['log10_k'].values)
    exp3 = eval_s2(y_te_log3, s2_exp3.predict(np.c_[Xa_te3, pn_te3]))

    # ── Print table ──────────────────────────────────────────────
    col_w = [6, 12, 23, 12, 11, 11]
    hdr4  = (f"  {'Exp':<{col_w[0]}}  {'Split':<{col_w[1]}}  {'Model':<{col_w[2]}}"
             f"  {'S1 metric':<{col_w[3]}}  {'R²(log-k)':>{col_w[4]}}"
             f"  {'R²(raw-k)':>{col_w[5]}}")
    print(hdr4)
    print("  " + SEP2)

    table_rows = [
        ('Exp1', 'Random',       'ExtraTreesRegressor',   'N/A',
         exp1_etr[0], exp1_etr[1]),
        ('Exp1', 'Random',       'RandomForestRegressor', 'N/A',
         exp1_rf[0],  exp1_rf[1]),
        ('Exp2', 'GroupShuffle', 'ExtraTreesRegressor',   'N/A',
         exp2_etr[0], exp2_etr[1]),
        ('Exp2', 'GroupShuffle', 'RandomForestRegressor', 'N/A',
         exp2_rf[0],  exp2_rf[1]),
        ('Exp3', 'GroupShuffle', 'ETC → XGBRegressor',
         f'F1={s1_mf1_3:.3f}',   exp3[0],    exp3[1]),
    ]

    for exp, spl, mdl, s1m, r2l, r2r in table_rows:
        row = (f"  {exp:<{col_w[0]}}  {spl:<{col_w[1]}}  {mdl:<{col_w[2]}}"
               f"  {s1m:<{col_w[3]}}  {r2l:>{col_w[4]}.4f}"
               f"  {r2r:>{col_w[5]}.4f}")
        print(row)

    # ── Effects (using ETR for Exp1 vs Exp2 consistency) ─────────
    leak_eff = exp1_etr[0] - exp2_etr[0]   # same model, diff split
    arch_eff = exp2_etr[0] - exp3[0]        # same split, diff architecture
    total    = exp1_etr[0] - exp3[0]

    print()
    print(f"  Leakage effect   (Exp1-ETR → Exp2-ETR):  ΔR²(log-k) = {leak_eff:+.4f}")
    print(f"  Architecture eff (Exp2-ETR → Exp3):       ΔR²(log-k) = {arch_eff:+.4f}")
    print(f"  Total            (Exp1-ETR → Exp3):       ΔR²(log-k) = {total:+.4f}")
    print()
    print("  Interpretation:")
    print(f"    Leakage inflated R² by {leak_eff:+.4f} "
          f"({'positive' if leak_eff > 0 else 'negative'} = "
          f"{'inflated by random split' if leak_eff > 0 else 'no inflation observed'})")
    print(f"    Full pipeline (ETC→XGB, 19-feat) "
          f"{'improves over' if arch_eff < 0 else 'is worse than'} "
          f"baseline regressor by {abs(arch_eff):.4f} R²")

    lines4 = [
        "FIX 4 — Leakage Isolation",
        "=" * 55, "",
        "Exp1: Random train_test_split  | 15 features | direct regression",
        "Exp2: GroupShuffleSplit        | 15 features | direct regression",
        "Exp3: GroupShuffleSplit        | 19 features | ETC → XGB pipeline",
        "",
        hdr4, "  " + SEP2,
    ]
    for exp, spl, mdl, s1m, r2l, r2r in table_rows:
        lines4.append(
            f"  {exp:<{col_w[0]}}  {spl:<{col_w[1]}}  {mdl:<{col_w[2]}}"
            f"  {s1m:<{col_w[3]}}  {r2l:>{col_w[4]}.4f}"
            f"  {r2r:>{col_w[5]}.4f}"
        )
    lines4 += [
        "",
        f"  Leakage effect   (Exp1-ETR → Exp2-ETR): ΔR²(log-k) = {leak_eff:+.4f}",
        f"  Architecture eff (Exp2-ETR → Exp3):      ΔR²(log-k) = {arch_eff:+.4f}",
        f"  Total            (Exp1-ETR → Exp3):      ΔR²(log-k) = {total:+.4f}",
        "",
        "  NOTE: positive leakage effect = random split inflated R² (test",
        "  alloys overlap with training alloys → memorisation, not generalisation).",
        "  Architecture effect: improvement from full 2-stage pipeline + 19 feats",
        "  over direct regression on 15 raw features with group-aware split.",
    ]
    write_txt('fix4.txt', lines4)
    print("\n[Fix 4 DONE]")

except Exception as exc:
    print(f"[Fix 4 FAILED]: {exc}")
    import traceback; traceback.print_exc()

# ─────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("ALL DONE. Files: fix1.txt  fix2.txt  fix2.png  fix3.txt  fix4.txt")
print(SEP)
