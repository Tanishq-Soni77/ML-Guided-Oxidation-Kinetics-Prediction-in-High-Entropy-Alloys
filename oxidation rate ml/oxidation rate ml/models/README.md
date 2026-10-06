# Trained models

Trained model checkpoints (`.pkl`) are **not** committed to this repo (see `.gitignore`) to keep it
lightweight and diff-friendly. This folder is a placeholder — running the pipeline scripts in
[`src/`](../src) regenerates them here:

| File | Produced by |
|---|---|
| `model_xgb.pkl`, `model_rf.pkl`, `model_et.pkl` | `src/stage1_classification/task3_xgb.py`, `task4_rf_et.py` |
| `model_xgb_3class.pkl`, `model_rf_3class.pkl`, `model_et_3class.pkl` | `src/stage1_classification/task3b_xgb_3class.py`, `task4b_rf_et_3class.py` |
| `model_xgb_s2.pkl`, `model_rf_s2.pkl`, `model_et_s2.pkl` | `src/stage2_regression/s2_task2_xgb.py`, `s2_task3_rf_et.py` |

See the top-level [README](../README.md#how-to-run-it) for the full run order.
