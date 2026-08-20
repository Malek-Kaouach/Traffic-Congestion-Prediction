# STGTransformer — Resilient Urban Mobility

> **A Multi-Modal Spatiotemporal Transformer–GNN Pipeline with SHAP-Style Explainability for Real-Time Traffic Congestion Prediction**

**Authors:** Peter Yacoub · Mohamed Malek Kaouach  
**Supervisor:** Prof. Mohammed Deriche  
**Institution:** Ajman University — AIRC 
**GitHub:** [PeterYNY/stgtransformer](https://github.com/PeterYNY/stgtransformer)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Results at a Glance](#2-results-at-a-glance)
3. [Repository Structure](#3-repository-structure)
4. [Local Project Data Layout](#4-local-project-data-layout)
5. [Prerequisites](#5-prerequisites)
6. [Dataset Setup](#6-dataset-setup)
7. [Phase-by-Phase Execution Guide](#7-phase-by-phase-execution-guide)
   - [Phase 1 — Data Download](#phase-1--data-download)
   - [Phase 2 & 3 — Preprocessing & Baselines](#phase-2--3--preprocessing--baselines)
   - [Phase 4 — Model Training (STGTransformer v2)](#phase-4--model-training-stgtransformer-v2)
   - [Phase 5 — Ablation & SHAP](#phase-5--ablation--shap)
   - [Phase 6 — Deployment](#phase-6--deployment)
8. [Quick Restart Guide](#8-quick-restart-guide)
9. [Model Architecture](#9-model-architecture)
10. [Key Results](#10-key-results)
11. [Citation](#11-citation)

---

## 1. Project Overview

STGTransformer combines:
- **Bidirectional Diffusion GNN** (2 layers) for spatial road network modeling
- **Temporal Transformer** (4 attention heads, 2 encoder layers, `d_model=128`) for long-range sequence reasoning
- **Multi-modal fusion** of traffic speed, cyclical time-of-day/day-of-week encodings, weather, and accident context (31 engineered features, including one-hot weather categories)
- **Gradient-based feature attribution** (input × gradient, computed per test sample) to explain which features drive each prediction

**Datasets used:**
- [PEMS-BAY](https://zenodo.org/record/5724362) — 325 highway sensors, SF Bay Area, Jan–Jun 2017
- [US-Accidents](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) — 7.7M countrywide accident records (Kaggle), filtered to ~35.6K Bay Area 2017 records

**Key finding:** traffic speed and time-of-day dominate the model's predictions — they account for **90.6%** of total feature attribution, versus **7.0%** for weather and **2.4%** for accident context. Consistent with this, an ablation study shows that adding weather + accident features does **not** improve accuracy: the Speed + Time-only model beats the full multi-modal model by **2–5% MAE** at every horizon. Multi-modal fusion remains a genuine architectural contribution and the explainability pipeline correctly surfaces this result — but it is not a validated accuracy win on this dataset.

---

## 2. Results at a Glance

| Model | 15-min MAE | 30-min MAE | 60-min MAE |
|---|---|---|---|
| Historical Average | 2.93 mph | 3.11 mph | 3.54 mph |
| ARIMA | 2.98 mph | 2.99 mph | 3.03 mph |
| LSTM (no spatial) | 1.47 mph | 1.97 mph | 2.61 mph |
| **STGTransformer v2 (ours)** | **1.46 mph** | **1.88 mph** | **2.38 mph** |

- Beats LSTM at every horizon (spatial modeling helps).
- Beats STGCN (2018, published) at 60 min: 2.38 vs 2.49 mph.
- Within ~6% of DCRNN (2018, published) at 15 min: 1.46 vs 1.38 mph.

**Masked MAPE:** **3.07% / 4.25% / 5.69%** at 15/30/60 min · **Best checkpoint:** `stgt_v2_continued_best.pt` (epoch 88 of 100, val_loss = 0.0763, 641,740 parameters)

---

## 3. Repository Structure

```
stgtransformer/
│
├── Phase_1.ipynb                      # Data download from Kaggle + PEMS-BAY verification
├── Phase2&3.ipynb                     # Preprocessing, EDA, baseline models (HA, ARIMA, LSTM)
├── run_before_phase4.ipynb            # Pre-Phase-4 environment/data sanity checks
├── Phase_4.ipynb                      # STGTransformer v1 (d_model=64) — superseded
├── Phase_4_STGTransformer_v2.ipynb    # STGTransformer v2 (d_model=128) — current best model
├── Phase_5.ipynb                      # Ablation study + gradient-based explainability
├── Phase6.ipynb                       # Deployment: FastAPI + Docker + Streamlit
│
├── traffic_project/                   # All data, checkpoints, results (see Section 4)
├── README.md                          # This file
└── LICENSE
```

> **Note:** the pipeline now runs entirely **locally** against `./traffic_project/` — no Google Drive/Colab mounting is required. All notebooks are self-contained and can be run independently as long as `./traffic_project/` has the expected upstream files.

---

## 4. Local Project Data Layout

All notebooks read from and write to `./traffic_project/` (relative to the repo root):

```
traffic_project/
│
├── 📦 raw_data/
│   ├── pems-bay.h5                    # PEMS-BAY speed data (52,116 × 325)
│   ├── pems-bay-meta.h5               # Sensor metadata (lat/lon)
│   ├── adj_mx_bay.pkl                 # Raw adjacency matrix
│   ├── US_Accidents_March23.csv       # Full US-Accidents dataset (Kaggle download)
│   └── us_accidents_bay_2017.csv      # Filtered Bay Area 2017 records
│
├── 🔧 processed_data/
│   ├── merged_dataset_full.parquet    # 16,937,700 rows × 16 cols (weather + accident merge)
│   ├── data_3d.npy                    # 3D array (52116, 325, 31) — model input
│   ├── adj_tensor.pt                  # PyTorch adjacency tensor
│   ├── scaler.pkl                     # StandardScaler fitted on train only
│   ├── config.json                    # Dataset/model config (n_features=31, batch_size, split indices)
│   ├── sensors_list.json / timestamps_list.json
│   └── weather_map.json               # Weather-condition → one-hot column mapping
│
├── 🤖 checkpoints/
│   ├── stgt_v2_continued_best.pt      # ✅ Best model — USE THIS (epoch 88/100, val_loss=0.0763)
│   ├── stgt_best.pt                   # STGTransformer v1 (d_model=64) — superseded
│   ├── stgt_train_losses.npy / stgt_val_losses.npy
│   └── lstm_best.pt                   # LSTM baseline checkpoint
│
├── 📊 results/
│   ├── all_results.json               # Model comparison (MAE)
│   ├── baseline_results.json          # HA + ARIMA results
│   ├── proper_ablation_results.json   # Ablation study results (3 feature-set variants)
│   ├── shap_values.npy                # Gradient-attribution values (15-min, 200 samples)
│   ├── shap_importance.png            # Feature importance bar chart
│   ├── stgt_test_preds.npy / stgt_test_trues.npy
│   └── lstm_test_preds.npy / lstm_test_trues.npy / ha_table.npy
│
├── 🧩 src/
│   ├── dataset.py                     # TrafficDataset (shared sliding-window loader)
│   ├── models.py                      # SpatioTemporalTransformer + DiffusionConvLayer
│   └── utils.py
│
└── 🚀 deployment/
    ├── model.py                       # Model class definition (loads checkpoint)
    ├── preprocess.py                  # Input preprocessing
    ├── main.py                        # FastAPI app (/predict endpoint)
    ├── requirements.txt / Dockerfile / .dockerignore
    ├── model_files/                   # Copied checkpoint + config for serving
    └── dashboard/
        └── app.py                     # Streamlit dashboard (dark glassmorphism UI)
```

---

## 5. Prerequisites

### Environment
All notebooks run **locally** in a Python virtual environment (a `.venv` is used in this repo). Model training in Phase 4 was run on an **RTX 4060 8GB** GPU; a CUDA GPU is recommended but not required (CPU works, just slower).

### Python packages
Each notebook installs/imports its own dependencies. The core stack is:

```
torch
numpy
pandas
scikit-learn
pyarrow
tables
h5py
matplotlib
python-dotenv
kaggle
fastapi
uvicorn
pydantic
streamlit
plotly
requests
```

### Kaggle API token
Phase 1 requires a Kaggle API token to download US-Accidents. Get yours from **[kaggle.com → Account → API → Create New Token](https://www.kaggle.com/settings)**, then put your credentials in a `.env` file at the repo root:

```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

Phase 1's setup cell calls `load_dotenv()` followed by `kaggle.api.authenticate()`.

---

## 6. Dataset Setup

### PEMS-BAY
Download from Zenodo and place manually into `./traffic_project/raw_data/`:

```
https://zenodo.org/record/5724362
```

Files needed:
- `pems-bay.h5` — speed data
- `pems-bay-meta.h5` — sensor metadata
- `adj_mx_bay.pkl` — adjacency matrix

### US-Accidents
Downloaded automatically in Phase 1 via the Kaggle API.

```
Dataset: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
Size: ~3GB (full dataset — Phase 1 filters to ~35.6K Bay Area 2017 records)
```

---

## 7. Phase-by-Phase Execution Guide

> **Important:** the pipeline runs entirely against `./traffic_project/` on local disk. No Drive-mount step is needed — just make sure you're running notebooks from the repo root.

---

### Phase 1 — Data Download

**Notebook:** `Phase_1.ipynb`  
**Purpose:** Authenticate with Kaggle, download US-Accidents, filter to Bay Area 2017, verify PEMS-BAY files, and produce the merged 16.9M-row corpus.  
**Runtime:** ~15–25 minutes (weather + accident-context merges are the slow steps)

**Steps:**
1. Create a `.env` file with your `KAGGLE_USERNAME` / `KAGGLE_KEY` (see [Section 5](#5-prerequisites))
2. Open `Phase_1.ipynb` and run all cells top to bottom
3. Manually upload the three PEMS-BAY files into `./traffic_project/raw_data/` before the verification cell

**What happens:**
- Downloads and filters US-Accidents to 35,645 Bay Area 2017 records
- Stage 1: nearest-in-time weather join (±4h) → **97.8% weather coverage**
- Stage 2: accident context window (5km radius, 60-min lookback) → `acc_count_60min`, `acc_max_severity`, `acc_mins_since`
- Saves `merged_dataset_full.parquet` — 16,937,700 rows × 16 columns

---

### Phase 2 & 3 — Preprocessing & Baselines

**Notebook:** `Phase2&3.ipynb`  
**Purpose:** Build the 3D model-ready array and run baseline models (HA, ARIMA, LSTM).  
**Runtime:** ~30–60 minutes

**Steps:**
1. Ensure Phase 1 is complete
2. Run all cells top to bottom

**What happens:**
- Adds cyclical time features and one-hot-encodes weather condition
- Builds `data_3d.npy` — shape **(52116, 325, 31)**, the final model input (31 features: speed, 5 time features, 4 weather numerics, 3 accident features, ~18 one-hot weather categories)
- Fits `scaler.pkl` (StandardScaler on train split only) and builds `adj_tensor.pt`
- Trains and evaluates Historical Average, ARIMA, and LSTM baselines

**Output files saved to `./traffic_project/`:**
```
processed_data/data_3d.npy
processed_data/adj_tensor.pt
processed_data/scaler.pkl
processed_data/config.json
processed_data/sensors_list.json
processed_data/timestamps_list.json
processed_data/merged_dataset_full.parquet
results/baseline_results.json
checkpoints/lstm_best.pt
results/lstm_test_preds.npy
results/lstm_test_trues.npy
```

---

### Phase 4 — Model Training (STGTransformer v2)

**Notebook:** `Phase_4_STGTransformer_v2.ipynb`  
**Purpose:** Train the current best STGTransformer (`d_model=128`) and evaluate on the test set.  
**Runtime:** ~2–3 hours for the full 100-epoch run on an RTX 4060 8GB GPU  

> `Phase_4.ipynb` (v1, `d_model=64`) is kept for reference but is **superseded** — it lost to the LSTM baseline. All current results come from `Phase_4_STGTransformer_v2.ipynb`.

**Cell guide:**

| Cell | Purpose |
|---|---|
| 1 | Setup: imports, load data + adjacency |
| 2 | Dataset & DataLoaders |
| 4 | Temporal Transformer encoder definition |
| 5 | Full model: `SpatioTemporalTransformer` (Diffusion GNN + Transformer) |
| 6 | Sanity-check forward pass |
| 7 | Training loop with checkpointing (50 epochs) |
| — | Continued run: resumes from best checkpoint, epochs 50 → 100 |
| 8 | Evaluate on test set → MAE / RMSE / MAPE |
| 9 | Full comparison table vs. baselines |

**Key hyperparameters:**
```python
d_model      = 128
n_heads      = 4          # kept fixed to isolate the d_model change from v1
n_gnn_layers = 2
n_tf_layers  = 2
dropout      = 0.1
seq_len      = 12         # input_steps
pred_steps   = 12         # 5-min steps out to 60 min; 15/30/60-min horizons read at steps 2/5/11
batch_size   = 32         # epochs 1-50; continued run (51-100) used batch_size=64 for speed
lr           = 3e-4
optimizer    = Adam(weight_decay=1e-4)
scheduler    = ReduceLROnPlateau(factor=0.5, patience=4)
loss         = HuberLoss(delta=1.0)
grad_clip    = 1.0
total_epochs = 100         # continued from a 50-epoch run whose best val_loss was 0.0775
```

**Output files saved to `./traffic_project/`:**
```
checkpoints/stgt_v2_continued_best.pt   ← best checkpoint (use this)
checkpoints/stgt_train_losses.npy
checkpoints/stgt_val_losses.npy
results/stgt_test_preds.npy
results/stgt_test_trues.npy
results/all_results.json
```

> ✅ **Best checkpoint:** `stgt_v2_continued_best.pt` — epoch 88 of 100, val_loss = 0.0763, 641,740 parameters. Epochs 89–100 bought nothing further (val loss plateaued at 0.0763–0.0766); the model was effectively converged by ~epoch 85–88.

---

### Phase 5 — Ablation & SHAP-Style Explainability

**Notebook:** `Phase_5.ipynb`  
**Purpose:** Ablation study (per-modality contribution) and gradient-based feature attribution (15-min horizon), loading `stgt_v2_continued_best.pt`.  
**Runtime:** ~35–45 minutes (3 ablation variants × 20 epochs) + a few minutes for attribution (200 samples, GPU)

**Cell guide:**

| Cell | Purpose |
|---|---|
| 1 | Setup: load data, adjacency, config |
| 2 | Dataset & DataLoaders |
| 3 | Model definition, load `stgt_v2_continued_best.pt` |
| 4 | Ablation helpers + run 3 variants |
| — | Gradient × input attribution (200 test samples, 15-min horizon) |
| 6 | Feature importance charts |
| 7 | Final summary |

**Ablation variants trained (20 epochs each, from scratch):**
1. Speed + Time only (4 features)
2. + Weather (9 features)
3. Full — Speed + Time + Weather + Accidents (31 features)

**Output files saved to `./traffic_project/`:**
```
results/proper_ablation_results.json
results/shap_values.npy
results/shap_importance.png
```

---

### Phase 6 — Deployment

**Notebook:** `Phase6.ipynb`  
**Purpose:** Build and launch the FastAPI backend + Streamlit dashboard for live traffic prediction.  
**Runtime:** ~5 minutes setup, dashboard runs indefinitely

**Run order:**

```
First time (full setup):
Cell 1 → Cell 2 → Cell 3 → Cell 4 → Cell 5 → Cell 6 → Cell 7 → Cell 8 → Cell 9 → Cell 10 → Cell 11

After restart (relaunch only):
Cell 1 → Cell 8 → Cell 10 → Cell 11
```

**Cell guide:**

| Cell | Purpose |
|---|---|
| 1 | Setup, set paths, create deployment folders |
| 2 | Write `model.py` — `SpatioTemporalTransformer` class (`d_model=128`, `n_heads=4`) |
| 3 | Write `preprocess.py` — input preprocessing |
| 4 | Write `main.py` — FastAPI `/predict` endpoint |
| 5 | Write `requirements.txt` and `Dockerfile` |
| 6 | Write Streamlit `app.py` — dark glassmorphism dashboard |
| 7 | Copy model files to `deployment/model_files/` |
| 8 | Start FastAPI server (background process) |
| 9 | Test `/predict` endpoint with sample JSON |
| 10 | Start Streamlit dashboard |
| 11 | Get local/public dashboard URL |
| 12 | Build Docker image (optional) |
| 13 | Deployment summary |

Checkpoint resolution in `model.py` / `main.py` prefers, in order: `stgt_v2_continued_best.pt` → `stgt_v2_best.pt` → `stgt_best.pt`. Both known `d_model=128` checkpoints use `n_heads=4` — this is hardcoded in `model.py` rather than inferred from `d_model`, since the state-dict shape doesn't reveal head count and an earlier auto-detection heuristic guessed `n_heads=8` incorrectly.

**Using the dashboard:**
1. Run Cell 11 to get the dashboard URL
2. Open the URL in your browser
3. In the sidebar: select Sensor ID, Hour of Day, Day of Week, Weather Condition, Nearby Accidents
4. Click **Predict** to get 15/30/60-min speed forecasts
5. View the live congestion map and Speed Forecast Timeline

**Testing the API directly:**
```python
import requests

payload = {
    "sensor_id": 60,
    "hour": 8,
    "day_of_week": 3,
    "weather_condition": "Fog",
    "nearby_accidents": 3
}

response = requests.post("http://localhost:8000/predict", json=payload)
print(response.json())
# {"15min_mph": 42.86, "30min_mph": 45.21, "60min_mph": 52.26, "congestion": "Medium"}
```

> ⚠️ The FastAPI server and Streamlit dashboard run as background processes. Re-run `Cell 1 → Cell 8 → Cell 10 → Cell 11` to restart after a kernel restart.

> ⚠️ Docker image build (Cell 12) is optional — it requires ~5GB disk space and a long build time. The Streamlit dashboard works without Docker.

---

## 8. Quick Restart Guide

Use this table when resuming after a kernel/session restart:

| Goal | Notebook | Cells to run |
|---|---|---|
| Re-run full training | `Phase_4_STGTransformer_v2.ipynb` | 1 → 2 → 4 → 5 → 6 → 7 → 8 → 9 |
| Ablation only | `Phase_5.ipynb` | 1 → 2 → 3 → 4 |
| Attribution / feature importance only | `Phase_5.ipynb` | 1 → 2 → 3 → 6 → 7 |
| Relaunch dashboard | `Phase6.ipynb` | 1 → 8 → 10 → 11 |

---

## 9. Model Architecture

```
Input X  (B, T=12, N=325, F=31)
    │
    ▼
Input Projection  ──  Linear: F=31 → d=128
    │
    ▼
Bidirectional Diffusion GNN × 2 layers (2-hop diffusion each)
    ├── Forward diffusion:   H_fwd = ReLU(A_fwd · H · W_fwd)
    ├── Backward diffusion:  H_bwd = ReLU(A_bwd · H · W_bwd)
    └── Spatial Gate g:      H_fused = g⊙H_gnn + (1-g)⊙H_orig
    │
    ▼
Temporal Transformer
    ├── 4 attention heads
    ├── 2 encoder layers
    └── Learnable positional embeddings
    │
    ▼
Prediction Head  ──  FC(128→64) → ReLU → Dropout(0.1) → FC(64→12)
    │
    ▼
Output  (B, H=12, N=325)  →  Speed at 15 / 30 / 60 min (steps 2/5/11) for all 325 sensors
```

**Total parameters:** 641,740  
**Training:** 100 epochs total (50 + 50 continued), RTX 4060 8GB GPU  
**Best checkpoint:** epoch 88, val_loss = 0.0763  
**Loss:** Huber (δ=1.0)  
**Optimizer:** Adam (weight_decay=1e-4) + `ReduceLROnPlateau` (factor=0.5, patience=4)

---

## 10. Key Results

### Performance (mph) — `stgt_v2_continued_best.pt`, test set

| Metric | 15 min | 30 min | 60 min |
|---|---|---|---|
| MAE | 1.46 | 1.88 | 2.38 |
| RMSE | 3.03 | 4.07 | 5.08 |
| Masked MAPE | 3.07% | 4.25% | 5.69% |

### Comparison vs. published models (MAE, mph)

| Model | 15 min | 30 min | 60 min |
|---|---|---|---|
| Historical Average | 2.93 | 3.11 | 3.54 |
| ARIMA | 2.98 | 2.99 | 3.03 |
| LSTM (no spatial) | 1.47 | 1.97 | 2.61 |
| **STGTransformer v2 (ours)** | **1.46** | **1.88** | **2.38** |
| DCRNN (2018, published) | 1.38 | 1.74 | 2.07 |
| STGCN (2018, published) | 1.36 | 1.81 | 2.49 |
| Graph WaveNet (2019, published) | 1.30 | 1.63 | 1.95 |

- Beats STGCN at 60 min (2.38 vs 2.49 mph); within ~6% of DCRNN at 15 min; beats LSTM at all horizons.

### Ablation study (multi-modal fusion — corrected finding)

Adding weather + accident features does **not** improve accuracy on this dataset. `Speed + Time only` scores best across all horizons; the `Full` multi-modal variant is **2–5% worse**:

| Variant | 15 min | 30 min | 60 min |
|---|---|---|---|
| Speed + Time only | 0.1579 | 0.2046 | 0.2622 |
| Speed + Time + Weather | 0.1571 | 0.2052 | 0.2628 |
| Full (Speed + Time + Weather + Accidents) | 0.1659 | 0.2133 | 0.2682 |

*(normalized-scale MAE; multi-modal gain vs. Speed+Time-only: −5.1% / −4.2% / −2.3%, i.e. worse)*

### Feature attribution (gradient × input, 15-min horizon, 200 test samples)

| Rank | Feature | Mean \|attribution\| | Group |
|---|---|---|---|
| 1 | Traffic Speed | 0.0562 | Traffic & Time |
| 2 | Weather = Clear | 0.0167 | Weather |
| 3 | Hour of Day (cos) | 0.0159 | Traffic & Time |
| 4 | Hour of Day (sin) | 0.0126 | Traffic & Time |
| 5 | Weather = Overcast | 0.0094 | Weather |
| 6 | Is Weekend | 0.0066 | Traffic & Time |
| 7 | Accident Severity | 0.0007 | Accident |
| 8 | Accident Count (60 min) | 0.0006 | Accident |

**Group breakdown:** Traffic & Time = **90.6%**, Weather = **7.0%**, Accident = **2.4%** of total attribution.

**Key finding:** traffic speed and time-of-day dominate the model's predictions; weather and accident signal is weak and noisy at this granularity — consistent with the ablation result above. Real-time incident feeds and weather stations both contribute little marginal predictive value once recent traffic speed and time-of-day are known.

---

## 11. Citation

If you use this code or results in your work, please cite:

```bibtex
@article{yacoub2025stgtransformer,
  title     = {Resilient Urban Mobility: A Multi-Modal Spatiotemporal
               Transformer--GNN Pipeline with Explainability
               for Real-Time Traffic Congestion Prediction},
  author    = {Yacoub, Peter and Kaouach, Mohamed Malek},
  journal   = {MAI603: Machine Learning, Ajman University},
  year      = {2026},
  note      = {Supervised by Prof. Mohammed Deriche, AIRC}
}
```



---

*Built with PyTorch, FastAPI, and Streamlit.*  
*PEMS-BAY dataset courtesy of [Li et al., ICLR 2018](https://arxiv.org/abs/1707.01926).*  
*US-Accidents dataset courtesy of [Moosavi et al., 2019](https://arxiv.org/abs/1906.05409).*
