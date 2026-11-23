# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Remaining Useful Life (RUL) prediction for turbofan engines using NASA's C-MAPSS dataset (`CMAPSSData/`, see its `readme.txt` for column meanings). An LSTM regressor (TensorFlow/Keras) predicts cycles remaining before failure from sliding windows of sensor readings. Not a git repo; no build, lint or test tooling. All work happens in Jupyter notebooks.

## Environment

- `venv/` (Python 3.11) only contains TensorFlow 2.21 / Keras 3. pandas, scikit-learn and matplotlib are not installed there, so the notebooks are run with a different kernel (`preprocess.ipynb` pip-installs tensorflow into whatever kernel is active).
- To run a notebook headlessly: `jupyter nbconvert --to notebook --execute <name>.ipynb --inplace` (requires jupyter in the active environment).

## Pipeline (run in this order)

1. **`preprocess.ipynb`**: the main notebook. Loads all four FD00x subsets, but then overrides to **FD001 only** (`rul = rulfd001`, etc.). Steps:
   - Drops near-constant columns `[5, 9, 10, 14, 20, 22, 23]`; renames cols 0/1 to `engine`/`cycle`. Remaining columns keep their **integer** names from the raw file.
   - Training target: piecewise-linear RUL capped at 125 (`min(125, max_cycle - cycle)`).
   - `GroupShuffleSplit` by engine (80/20, seed 42) so no engine appears in both train and val.
   - `StandardScaler` fit on the train split only, then applied to val and test.
   - Writes `test.csv` (scaled test features) and `rul.csv` (true test RUL, written *before* the 125 cap is applied later in the notebook).
   - `create_windows` builds 30-cycle windows → `X` shape `(n, 30, 17)`, target = RUL at window end.
   - `run_experiment` trains configs from the `experiments` list, writes `rul_experiments.csv`, and the chosen model is saved as `single_lstm_64.keras`.
2. **`predict.ipynb`**: loads `single_lstm_64.keras`, `test.csv`, `rul.csv`, takes the last window per test engine, and reports RMSE / MAE / R² / NASA score plus plots.
3. **`experiment_eval.ipynb`**: small analysis of `rul_experiments.csv`.

## Things to know

- **NASA scoring function** (`nasa_loss`) is both the training loss and an evaluation metric: asymmetric exponential penalty, `exp(-d/13)-1` for early predictions and `exp(d/10)-1` for late ones (`d = pred - true`). It is a custom object, so loading the model needs `custom_objects={'nasa_loss': nasa_loss}`. The function is duplicated in `preprocess.ipynb` and `predict.ipynb`; keep them in sync.
- `run_experiment` accepts a `loss` argument but **ignores it**: every model compiles with `nasa_loss`. The `loss` column in `rul_experiments.csv` (`mse`, Huber) is just a label.
- Early stopping monitors `val_last_window`, a custom callback (`LastWindowScore`) that scores the NASA metric on 5 random windows per validation engine, not `val_loss`.
- Feature columns: `[2, 3, 4, 6, 7, 8, 11, 12, 13, 15, 16, 17, 18, 19, 21, 24, 25]` (3 operating settings + 14 sensors). After the CSV round-trip, column names become strings, so `predict.ipynb` converts them with `str(...)`.
- Window length (30) and the RUL cap (125) are hardcoded in several places; change them everywhere at once.
- Several cells in `preprocess.ipynb` (test evaluation, cells referencing `windows_per_engine`) depend on variables that are never defined, so they fail when the notebook is run top to bottom. `predict.ipynb` is the working evaluation path.
