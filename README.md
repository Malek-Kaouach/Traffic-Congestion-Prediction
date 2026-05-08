# STGTransformer — Resilient Urban Mobility

> **A Multi-Modal Spatiotemporal Transformer–GNN Pipeline with SHAP Explainability for Real-Time Traffic Congestion Prediction**

**Authors:** Peter Yacoub · Mohamed Malek Kaouach  
**Supervisor:** Prof. Mohammed Deriche  
**Institution:** Ajman University — AIRC | MAI603: Machine Learning | May 2026  
**GitHub:** [PeterYNY/stgtransformer](https://github.com/PeterYNY/stgtransformer)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Results at a Glance](#2-results-at-a-glance)
3. [Repository Structure](#3-repository-structure)
4. [Google Drive Folder Structure](#4-google-drive-folder-structure)
5. [Prerequisites](#5-prerequisites)
6. [Dataset Setup](#6-dataset-setup)
7. [Phase-by-Phase Execution Guide](#7-phase-by-phase-execution-guide)
   - [Phase 1 — Data Download](#phase-1--data-download)
   - [Phase 2 & 3 — Preprocessing & Baselines](#phase-2--3--preprocessing--baselines)
   - [Phase 4 — Model Training](#phase-4--model-training)
   - [Phase 5 — Ablation & SHAP](#phase-5--ablation--shap)
   - [Phase 6 — Deployment](#phase-6--deployment)
8. [Quick Restart Guide](#8-quick-restart-guide)
9. [Model Architecture](#9-model-architecture)
10. [Key Results](#10-key-results)
11. [Citation](#11-citation)

---

## 1. Project Overview

STGTransformer is the first framework to combine:
- **Bidirectional Diffusion GNN** for spatial road network modeling
- **Temporal Transformer** (8 heads, 2 layers) for long-range sequence reasoning
- **Multi-modal fusion** of traffic speed, weather, and accident context (12 features)
- **SHAP explainability** revealing which features drive congestion predictions

**Datasets used:**
- [PEMS-BAY](https://zenodo.org/record/5724362) — 325 highway sensors, SF Bay Area, Jan–Jun 2017
- [US-Accidents](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) — 7.7M countrywide accident records (Kaggle)

**Key finding:** Multi-modal enrichment achieves competitive accuracy (masked MAPE 5.91% at 60 min) while processing 12 heterogeneous features — and SHAP analysis reveals accident features individually outrank meteorological features in predictive importance.

---

## 2. Results at a Glance

| Model | 15-min MAE | 30-min MAE | 60-min MAE |
|---|---|---|---|
| Historical Average | 2.93 mph | 3.11 mph | 3.54 mph |
| ARIMA | 2.98 mph | 2.99 mph | 3.03 mph |
| LSTM | 1.48 mph | 1.97 mph | 2.61 mph |
| **STGTransformer (ours)** | **1.48 mph** | **1.93 mph** | **2.48 mph** |
| STGTransformer (5 seeds) | 1.51 ± 0.02 | 1.97 ± 0.03 | 2.54 ± 0.07 |

**Masked MAPE** (speed > 5 mph): **3.10% / 4.35% / 5.91%** at 15/30/60 min

---

## 3. Repository Structure

```
stgtransformer/
│
├── Phase_1.ipynb              # Data download from Kaggle + PEMS-BAY
├── Phase2_Phase3.ipynb        # Preprocessing, EDA, baseline models (HA, ARIMA, LSTM)
├── Phase_4.ipynb              # STGTransformer training + multi-seed validation
├── Phase_55.ipynb             # Ablation study + SHAP explainability
├── Phase66.ipynb              # Deployment: FastAPI + Docker + Streamlit
│
├── README.md                  # This file
└── LICENSE
```

> **Note:** All data files, model checkpoints, and generated outputs are stored on Google Drive (not in this repository). See [Section 4](#4-google-drive-folder-structure) for the full Drive layout.

---

## 4. Google Drive Folder Structure

All notebooks read from and write to:

```
/content/drive/MyDrive/traffic_project/
│
├── 📦 Raw Data
│   ├── pems-bay.h5                    # PEMS-BAY speed data (52,116 × 325)
│   ├── pems-bay-meta.h5               # Sensor metadata (lat/lon/highway info)
│   ├── adj_mx_bay.pkl                 # Raw adjacency matrix
│   ├── US_Accidents_March23.csv       # Full US-Accidents dataset
│   └── us_accidents_bay_2017.csv      # Filtered Bay Area 2017 records
│
├── 🔧 Processed Data
│   ├── merged_dataset_full.parquet    # 16.9M rows × 16 features (Stage 1+2 merged)
│   ├── data_3d.npy                    # 3D array (52116, 325, 12) — model input
│   ├── adj_tensor.pt                  # PyTorch adjacency tensor
│   ├── scaler.pkl                     # StandardScaler fitted on train only
│   ├── config.json                    # Model hyperparameters
│   ├── sensors_list.json              # Sensor IDs
│   └── timestamps_list.json          # Timestamp index
│
├── 🤖 Model Checkpoints
│   ├── stgt_v2_best.pt                # ✅ Best model — USE THIS (epoch 50, val=0.0776)
│   ├── stgt_best.pt                   # Earlier checkpoint (epoch ~30)
│   └── lstm_best.pt                   # LSTM baseline checkpoint
│
├── 📊 Results & Outputs
│   ├── stgt_test_preds.npy            # STGTransformer test predictions
│   ├── stgt_test_trues.npy            # Test ground truth
│   ├── lstm_test_preds.npy            # LSTM test predictions
│   ├── lstm_test_trues.npy            # LSTM test ground truth
│   ├── ha_table.npy                   # Historical average lookup table
│   ├── baseline_results.json          # HA + ARIMA results
│   ├── all_results.json               # All model results
│   ├── final_results_mph_v2.json      # Final results converted to mph
│   ├── proper_ablation_results.json   # Ablation study results
│   ├── multi_seed_results.json        # 5-seed statistical significance results
│   ├── shap_values.npy                # SHAP values (15-min, 200 samples)
│   ├── stgt_v2_training_log.txt       # Full training log
│   ├── stgt_v2_train_losses.npy       # Training loss curve
│   └── stgt_v2_val_losses.npy         # Validation loss curve
│
├── 🖼️ Figures
│   ├── sensor_map.png                 # Geographic sensor distribution map
│   ├── pred_vs_gt.png                 # Prediction vs ground truth (1-week)
│   ├── shap_importance.png            # SHAP feature importance bar chart
│   ├── shap_feature_importance.png    # Alternative SHAP figure
│   ├── shap_direction_and_time.png    # SHAP direction analysis
│   └── shap_group_pie.png             # SHAP group breakdown pie chart
│
└── 🚀 Deployment
    └── deployment/
        ├── model.py                   # Model class definition
        ├── preprocess.py              # Input preprocessing
        ├── main.py                    # FastAPI app
        ├── requirements.txt           # Python dependencies
        ├── Dockerfile                 # Docker container definition
        └── dashboard/
            └── app.py                 # Streamlit dashboard
```

---

## 5. Prerequisites

### Environment
All notebooks run on **Google Colab** (free or Pro tier). A **T4 GPU** is recommended for Phase 4 training.

### Python packages
Each notebook installs its own dependencies in Cell 1. The core stack is:

```
torch >= 2.0
torch-geometric
numpy
pandas
scikit-learn
pyarrow
h5py
shap
matplotlib
contextily
pyproj
fastapi
uvicorn
streamlit
```

### Kaggle API token
Phase 1 requires a Kaggle API token to download US-Accidents. Get yours from:  
**[kaggle.com → Account → API → Create New Token](https://www.kaggle.com/settings)**

---

## 6. Dataset Setup

### PEMS-BAY
Download from Zenodo and upload to your Drive manually:

```
https://zenodo.org/record/5724362
```

Files needed:
- `pems-bay.h5` — speed data
- `pems-bay-meta.h5` — sensor metadata
- `adj_mx_bay.pkl` — adjacency matrix

Upload all three to `/content/drive/MyDrive/traffic_project/`

### US-Accidents
Downloaded automatically in Phase 1 via the Kaggle API.

```
Dataset: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
Size: ~3GB (full dataset — Phase 1 filters to ~35K Bay Area records)
```

---

## 7. Phase-by-Phase Execution Guide

> **Important:** Always mount Google Drive at the start of each session. Each notebook's Cell 1 does this automatically.

---

### Phase 1 — Data Download

**Notebook:** `Phase_1.ipynb`  
**Purpose:** Download US-Accidents from Kaggle and verify PEMS-BAY files exist on Drive.  
**Runtime:** ~5–10 minutes

**Steps:**

1. Open `Phase_1.ipynb` in Google Colab
2. In **Cell 1**, paste your Kaggle API token:
```python
os.environ['KAGGLE_API_TOKEN'] = "your_kaggle_token_here"
```
3. Run all cells top to bottom

**Output:**
- `US_Accidents_March23.csv` downloaded to Drive
- `us_accidents_bay_2017.csv` — filtered Bay Area 2017 records

> ⚠️ PEMS-BAY files (`pems-bay.h5`, `pems-bay-meta.h5`, `adj_mx_bay.pkl`) must be uploaded manually to Drive before proceeding.

---

### Phase 2 & 3 — Preprocessing & Baselines

**Notebook:** `Phase2_Phase3.ipynb`  
**Purpose:** Load and merge datasets, build the 3D model-ready array, run baseline models (HA, ARIMA, LSTM).  
**Runtime:** ~45–90 minutes (merge is the slow step)

**Steps:**

1. Open `Phase2_Phase3.ipynb` in Google Colab
2. Ensure Phase 1 is complete and Drive files exist
3. Run all cells top to bottom

**What happens:**
- Stage 1: Nearest-in-time weather join (≤50km, ≤4h) → 97.8% weather coverage
- Stage 2: Accident context window (5km radius, 60-min lookback)
- Outputs `merged_dataset_full.parquet` (16.9M rows × 16 features)
- Builds `data_3d.npy` (52116, 325, 12) — the final model input
- Trains and evaluates HA, ARIMA(1,1,1), and LSTM baselines
- Saves `scaler.pkl`, `adj_tensor.pt`, `baseline_results.json`

**Output files saved to Drive:**
```
data_3d.npy
adj_tensor.pt
scaler.pkl
config.json
sensors_list.json
timestamps_list.json
merged_dataset_full.parquet
baseline_results.json
lstm_best.pt
lstm_test_preds.npy
lstm_test_trues.npy
```

---

### Phase 4 — Model Training

**Notebook:** `Phase_4.ipynb`  
**Purpose:** Train the STGTransformer, evaluate on test set, run 5-seed statistical significance.  
**Runtime:** ~50 min (single run) | ~2.5 hours (multi-seed)  
**Recommended GPU:** T4

**Run order:**

```
First time (full training):
Cell 1 → Cell 2 → Cell 3 → Cell 4 → Cell 5 → Cell 6 → Cell 7 → Cell 8

After session restart (multi-seed only):
Cell 1 → Cell 2 → Cell 3 → Cell 8
```

**Cell guide:**

| Cell | Purpose |
|---|---|
| 1 | Mount Drive, imports, set paths |
| 2 | Build Dataset and DataLoaders |
| 3 | Define STGTransformer architecture |
| 4 | Training loop (50 epochs, Huber loss, Cosine Annealing) |
| 5 | Evaluate on test set → MAE / RMSE / MAPE |
| 6 | Convert results to mph, save figures |
| 7 | Save all results to JSON |
| 8 | Multi-seed validation (5 seeds × 30 epochs) — ~2.5 hours |

**Key hyperparameters:**
```python
d_model     = 128
n_heads     = 8
n_layers    = 2
seq_len     = 12
batch_size  = 16
lr          = 3e-4
epochs      = 50
loss        = HuberLoss(delta=1.0)
scheduler   = CosineAnnealingLR
```

**Output files saved to Drive:**
```
stgt_v2_best.pt               ← best checkpoint (use this)
stgt_test_preds.npy
stgt_test_trues.npy
stgt_v2_train_losses.npy
stgt_v2_val_losses.npy
stgt_v2_training_log.txt
stgt_v2_results.json
final_results_mph_v2.json
multi_seed_results.json       ← after Cell 8
```

> ✅ **Best checkpoint:** `stgt_v2_best.pt` — epoch 50, val_loss = 0.0776, 639,308 parameters

---

### Phase 5 — Ablation & SHAP

**Notebook:** `Phase_55.ipynb`  
**Purpose:** Ablation study (per-modality contribution), SHAP explainability (15-min horizon).  
**Runtime:** ~30 min (ablation) + ~15–20 min (SHAP on GPU) or ~4 hours (SHAP on CPU)

**Run order:**

```
First time:
Cell 1 → Cell 2 → Cell 3 → Cell 4 → Cell 5 → Cell 6 → Cell 7

After session restart (skip ablation, run SHAP only):
Cell 1 → Cell 2 → Cell 3 → Cell 5 → Cell 6 → Cell 7
```

**Cell guide:**

| Cell | Purpose |
|---|---|
| 1 | Mount Drive, imports, load data + adjacency |
| 2 | Build DataLoaders |
| 3 | Define model, load `stgt_v2_best.pt` |
| 4 | Ablation helper functions |
| 5 | Run ablation (3 variants × 25 epochs) |
| 6 | SHAP KernelExplainer (200 test samples, 15-min horizon) |
| 7 | Final summary + save all figures |

**Ablation variants trained:**
1. Speed + Time features only
2. + Weather (5 features)
3. + Accidents (full 12-feature model)

**Output files saved to Drive:**
```
proper_ablation_results.json
shap_values.npy
shap_importance.png
shap_feature_importance.png
shap_direction_and_time.png
shap_group_pie.png
sensor_map.png
pred_vs_gt.png
```

> ⚠️ SHAP KernelExplainer is computationally intensive (~160 seconds/sample on CPU). Run on GPU when possible or reduce `nsamples` to speed up.

---

### Phase 6 — Deployment

**Notebook:** `Phase66.ipynb`  
**Purpose:** Build and launch the Streamlit dashboard for live traffic prediction.  
**Runtime:** ~5 minutes setup, dashboard runs indefinitely

**Run order:**

```
First time (full setup):
Cell 1 → Cell 2 → Cell 3 → Cell 4 → Cell 5 → Cell 6 → Cell 7 → Cell 8 → Cell 9 → Cell 10 → Cell 11

After session restart (relaunch only):
Cell 1 → Cell 8 → Cell 10 → Cell 11
```

**Cell guide:**

| Cell | Purpose |
|---|---|
| 1 | Mount Drive, set paths, create deployment folders |
| 2 | Write `model.py` — STGTransformer class |
| 3 | Write `preprocess.py` — input preprocessing |
| 4 | Write `main.py` — FastAPI `/predict` endpoint |
| 5 | Write `requirements.txt` and `Dockerfile` |
| 6 | Write Streamlit `app.py` — dashboard |
| 7 | Copy model files to deployment folder |
| 8 | Start FastAPI server (background process) |
| 9 | Test `/predict` endpoint with sample JSON |
| 10 | Start Streamlit dashboard |
| 11 | Get public URL via localtunnel |
| 12 | Build Docker image (optional) |
| 13 | Deployment summary |

**Using the dashboard:**
1. Run Cell 11 to get the localtunnel URL
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

> ⚠️ The FastAPI server and Streamlit dashboard run as background processes inside Colab. They stop when the Colab session ends. Re-run `Cell 1 → Cell 8 → Cell 10 → Cell 11` to restart.

> ⚠️ Docker image build (Cell 12) is optional — it requires ~5GB disk space and a long runtime. The Streamlit dashboard works without Docker.

---

## 8. Quick Restart Guide

Use this table when resuming after a Colab session timeout:

| Goal | Notebook | Cells to run |
|---|---|---|
| Re-run full training | `Phase_4.ipynb` | 1 → 2 → 3 → 4 → 5 → 6 → 7 |
| Multi-seed only | `Phase_4.ipynb` | 1 → 2 → 3 → 8 |
| Ablation only | `Phase_55.ipynb` | 1 → 2 → 3 → 4 → 5 |
| SHAP only | `Phase_55.ipynb` | 1 → 2 → 3 → 5 → 6 → 7 |
| Relaunch dashboard | `Phase66.ipynb` | 1 → 8 → 10 → 11 |

---

## 9. Model Architecture

```
Input X  (B, T=12, N=325, F=12)
    │
    ▼
Input Projection  ──  Linear: F=12 → d=128
    │
    ▼
Bidirectional Diffusion GNN × 2
    ├── Forward diffusion:   H_fwd = ReLU(A_fwd · H · W_fwd)
    ├── Backward diffusion:  H_bwd = ReLU(A_bwd · H · W_bwd)
    └── Spatial Gate g:      H_fused = g⊙H_gnn + (1-g)⊙H_orig
    │
    ▼
Temporal Transformer
    ├── 8 attention heads
    ├── 2 encoder layers
    └── Learnable positional embeddings
    │
    ▼
Prediction Head  ──  FC → ReLU → Dropout(0.1) → FC
    │
    ▼
Output  (B, H=12, N=325)  →  Speed at 15 / 30 / 60 min for all 325 sensors
```

**Total parameters:** 639,308  
**Training:** 50 epochs, T4 GPU, ~62s/epoch  
**Loss:** Huber (δ=1.0) — reduces MAPE by 6.3% vs MSE  
**Optimizer:** Adam + Cosine Annealing (lr=3e-4 → 1e-5)

---

## 10. Key Results

### Performance (mph)

| Metric | 15 min | 30 min | 60 min |
|---|---|---|---|
| MAE | 1.48 | 1.93 | 2.48 |
| RMSE | 3.09 | 4.19 | 5.28 |
| Masked MAPE | 3.10% | 4.35% | 5.91% |

### Statistical Significance (5 random seeds)

| Horizon | MAE ± std |
|---|---|
| 15 min | 1.51 ± 0.02 mph |
| 30 min | 1.97 ± 0.03 mph |
| 60 min | 2.54 ± 0.07 mph |

### SHAP Feature Importance (15-min horizon)

| Rank | Feature | Mean \|SHAP\| | Group |
|---|---|---|---|
| 1 | Traffic Speed | 0.297 | Traffic |
| 2 | Hour of Day | 0.020 | Traffic |
| 3 | Is Weekend | 0.003 | Traffic |
| 4 | Accident Severity | 0.002 | Accident |
| 5 | Accident Count (60min) | 0.002 | Accident |
| 6 | Humidity | 0.001 | Weather |
| 7 | Visibility | 0.001 | Weather |

**Key finding:** Accident features (ranks 4–5) outrank weather features (ranks 6–7) individually — real-time incident feeds deliver more predictive value per sensor than distributed weather stations.

---

## 11. Citation

If you use this code or results in your work, please cite:

```bibtex
@article{yacoub2025stgtransformer,
  title     = {Resilient Urban Mobility: A Multi-Modal Spatiotemporal
               Transformer--GNN Pipeline with SHAP Explainability
               for Real-Time Traffic Congestion Prediction},
  author    = {Yacoub, Peter and Kaouach, Mohamed Malek},
  journal   = {MAI603: Machine Learning, Ajman University},
  year      = {2026},
  note      = {Supervised by Prof. Mohammed Deriche, AIRC}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with PyTorch, PyTorch Geometric, SHAP, FastAPI, and Streamlit.*  
*PEMS-BAY dataset courtesy of [Li et al., ICLR 2018](https://arxiv.org/abs/1707.01926).*  
*US-Accidents dataset courtesy of [Moosavi et al., 2019](https://arxiv.org/abs/1906.05409).*
