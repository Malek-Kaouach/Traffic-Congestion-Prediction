# Project Context — Traffic Congestion Prediction

**Course:** MAI603 — Machine Learning | Ajman University
**Last updated:** 2026-08-20

This file tracks project state across sessions: pipeline, model results, and
what was changed/fixed most recently. Read this before resuming work.

## Project overview

Predicts traffic speed on the **PEMS-BAY** road network (325 sensors, SF Bay
Area) at 15/30/60-minute horizons, fusing traffic speed history with weather
and accident data (from the US-Accidents Kaggle dataset, merged in Phase 1).

Notebook pipeline (run in order):

| Notebook | Purpose |
|---|---|
| `Phase_1.ipynb` | Downloads US-Accidents dataset from Kaggle, spatial-temporal merge → 16.9M-row corpus |
| `Phase2&3.ipynb` | Data loader & preprocessing → `traffic_project/processed_data/` (data_3d.npy, adj_tensor.pt, scaler.pkl, config.json) |
| `run_before_phase4.ipynb` | Pre-Phase-4 setup |
| `Phase_4.ipynb` | Original STGTransformer (v1), `d_model=64` |
| `Phase_4_STGTransformer_v2.ipynb` | STGTransformer v2, `d_model=128` — **current best model** |
| `Phase_5.ipynb` | Ablation study + SHAP explainability + final results summary |
| `Phase6.ipynb` | Deployment pipeline: FastAPI + Docker + Streamlit |

## Current best model

- **Architecture:** `SpatioTemporalTransformer` — diffusion-graph-conv (GNN) + Transformer encoder, defined in `Phase_4_STGTransformer_v2.ipynb` / mirrored in `Phase_5.ipynb` and `traffic_project/deployment/model.py`
- **Hyperparameters:** `d_model=128`, `n_heads=4`, `n_gnn_layers=2`, `n_tf_layers=2`, `dropout=0.1`, 641,740 params
- **Checkpoint file:** `traffic_project/checkpoints/stgt_v2_continued_best.pt` (epoch 88, val_loss=0.0763) — trained by continuing a prior 50-epoch run to 100 epochs total in Phase 4 v2.
- Other checkpoints present but superseded: `stgt_v2_best.pt`, `stgt_v2_last.pt`, `stgt_best.pt` (v1, d_model=64), `lstm_best.pt` (LSTM baseline).

**Important:** both the v1 (`d_model=64`) and v2 (`d_model=128`) checkpoints use **`n_heads=4`** — this was NOT `8` for d_model=128 despite what Phase6's old auto-detection logic assumed (see Fixes below).

## Phase 5 results (final, as of this session)

**Model comparison (MAE, mph):**

| Model | 15 min | 30 min | 60 min |
|---|---|---|---|
| Historical Average | 2.93 | 3.11 | 3.54 |
| ARIMA | 2.98 | 2.99 | 3.03 |
| LSTM (no spatial) | 1.47 | 1.97 | 2.61 |
| **STGTransformer (ours)** | **1.46** | **1.91** | **2.42** |
| DCRNN (2018, published) | 1.38 | 1.74 | 2.07 |
| STGCN (2018, published) | 1.36 | 1.81 | 2.49 |
| Graph WaveNet (2019, published) | 1.30 | 1.63 | 1.95 |

- Beats STGCN at 60 min (2.42 vs 2.49 mph); within 5.8% of DCRNN at 15 min; beats LSTM at all horizons.

**Ablation study (multi-modal fusion) — corrected finding:** adding weather +
accident features did **NOT** improve accuracy. `Speed+Time only` scored best
(MAE 0.1579 / 0.2046 / 0.2622 @ 15/30/60 min) vs. the `Full` multi-modal
variant (0.1659 / 0.2133 / 0.2682) — **2–5% worse** across all horizons. The
Key Findings cell previously claimed multi-modal fusion as a win; this was
false and has been corrected to flag it as a non-improvement (⚠️).

**SHAP feature attribution** (gradient × input, 200 test samples, 15-min
horizon) confirms this: **Traffic & Time = 90.6%**, Weather = 7.0%, Accident
= 2.4% of total attribution. Traffic speed and hour-of-day dominate; weather
and accident signal is weak.

**Genuine novel contributions:** multi-modal fusion architecture (even though
it didn't help accuracy here) and integrated SHAP explainability.

## Fixes made this session (2026-08-20)

### Phase_5.ipynb — Cell 7 (Final Summary)
- Corrected "Key Findings" to report the ablation study honestly (was
  previously claiming multi-modal fusion as a benefit; actual result is a
  2–5% MAE regression — see above).
- Added the SHAP group breakdown (90.6% / 7.0% / 2.4%) as a finding.
- Removed the "Files on Drive" section per user request (this repo runs
  locally, not on Drive — that block was leftover from a Colab-based
  version of the notebook).

### Phase6.ipynb — checkpoint path bugs fixed
The deployment notebook still hardcoded the old v1 checkpoint
(`stgt_best.pt`, d_model=64) as primary, with `stgt_v2_best.pt` as
fallback — it never looked for `stgt_v2_continued_best.pt`, the actual
best/final model from Phase 4 v2. Fixed across 5 cells:

- **Cell 0** (markdown header): added explicit `**Checkpoint:**
  stgt_v2_continued_best.pt (epoch 88, val_loss=0.0763)` line.
- **Cell 2** (local checkpoint existence check): now searches
  `['stgt_v2_continued_best.pt', 'stgt_v2_best.pt', 'stgt_best.pt']` in
  that order.
- **Cell 4** (`model.py` generator — `TrafficPredictor.__init__`): fixed a
  real bug where `ckpt_n_heads = 8 if ckpt_d_model == 128 else 4` would
  have loaded the d_model=128 checkpoint with the *wrong* number of
  attention heads (state_dict shapes don't depend on n_heads, so this would
  silently load and run with an incorrect head split rather than erroring).
  Both known checkpoints actually use `n_heads=4`, so this is now hardcoded
  with an explanatory comment.
- **Cell 8** (`main.py` generator — FastAPI startup): checkpoint filename
  resolution now prefers `stgt_v2_continued_best.pt` in the same 3-way
  fallback chain.
- **Cell 14** (copy runtime artifacts to `deployment/model_files/`): same
  3-way fallback chain.
- **Cell 27** (final pipeline summary): `deploy_files` inventory dict now
  checks for `model_files/stgt_v2_continued_best.pt` instead of the stale
  `stgt_best.pt`.

These notebooks are large enough (>25k tokens) that they can't be read by
standard file tools in one pass — edits were applied by loading/patching the
`.ipynb` JSON directly with a Python script rather than the normal
cell-based editor, to avoid truncation issues.

**Post-fix validation:** the user re-ran the affected Phase6 cells; the
regenerated `traffic_project/deployment/main.py` and `model.py` on disk
reflect the fix, and `traffic_project/deployment/model_files/
stgt_v2_continued_best.pt` now exists — confirming the corrected checkpoint
resolution and copy logic work end-to-end.

## Repo state / uncommitted changes (as of last check)

Not yet committed:
- `Phase6.ipynb` — checkpoint path fixes (above)
- `Phase_4_STGTransformer_v2.ipynb` — re-executed (adds/updates outputs)
- `Phase_5.ipynb` — Key Findings correction, Files-on-Drive removal
- `traffic_project/deployment/main.py`, `model.py` — regenerated from the
  fixed Phase6 cells
- New/untracked: `traffic_project/deployment/model_files/stgt_v2_continued_best.pt`

None of this has been committed — ask before committing/pushing.

## Known gotchas

- `Phase_4_STGTransformer_v2.ipynb`, `Phase_5.ipynb`, and `Phase6.ipynb` are
  each too large to `Read` in one call (>25k tokens); use `grep`/a small
  Python/json script to inspect or patch specific cells instead of the
  notebook-aware Read/NotebookEdit tools.
- The project runs **locally** (`./traffic_project/...`), not against Google
  Drive — some older cells/comments still reference "Drive" from an earlier
  Colab-based version; treat those as stale unless the code itself uses a
  Drive path.
