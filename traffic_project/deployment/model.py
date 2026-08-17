import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os

# ── Model Architecture ────────────────────────────────────────
class DiffusionConvLayer(nn.Module):
    def __init__(self, d_model, n_hops=2):
        super().__init__()
        self.fwd_linears = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(n_hops)
        ])
        self.bwd_linears = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(n_hops)
        ])
        self.out  = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, adj_fwd, adj_bwd):
        h_fwd = x
        for layer in self.fwd_linears:
            h_fwd = F.relu(layer(
                torch.einsum("nm, bmd -> bnd", adj_fwd, h_fwd)
            ))
        h_bwd = x
        for layer in self.bwd_linears:
            h_bwd = F.relu(layer(
                torch.einsum("nm, bmd -> bnd", adj_bwd, h_bwd)
            ))
        combined = torch.cat([h_fwd, h_bwd], dim=-1)
        return self.norm(self.out(combined) + x)


class SpatioTemporalTransformer(nn.Module):
    def __init__(self, n_features, d_model, n_heads, n_gnn_layers,
                 n_tf_layers, n_sensors, pred_steps, dropout=0.1):
        super().__init__()
        self.n_sensors  = n_sensors
        self.pred_steps = pred_steps
        self.d_model    = d_model
        INPUT_STEPS     = 12

        self.input_proj = nn.Linear(n_features, d_model)
        self.gnn_layers = nn.ModuleList([
            DiffusionConvLayer(d_model) for _ in range(n_gnn_layers)
        ])
        self.spatial_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.Sigmoid()
        )
        self.pos_embedding = nn.Parameter(
            torch.randn(1, INPUT_STEPS, d_model) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_tf_layers
        )
        self.tf_norm   = nn.LayerNorm(d_model)
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, pred_steps)
        )

    def forward(self, x, adj_fwd, adj_bwd):
        B, T, N, n_feat = x.shape
        x      = self.input_proj(x)
        x_orig = x.clone()
        x_flat = x.reshape(B * T, N, self.d_model)
        for gnn in self.gnn_layers:
            x_flat = gnn(x_flat, adj_fwd, adj_bwd)
        x_spatial = x_flat.reshape(B, T, N, self.d_model)
        gate = self.spatial_gate(torch.cat([x_spatial, x_orig], dim=-1))
        x    = gate * x_spatial + (1 - gate) * x_orig
        x    = x.permute(0, 2, 1, 3).reshape(B * N, T, self.d_model)
        x    = x + self.pos_embedding[:, :T, :]
        x    = self.transformer(x)
        x    = self.tf_norm(x[:, -1, :])
        pred = self.pred_head(x)
        pred = pred.reshape(B, N, self.pred_steps)
        pred = pred.permute(0, 2, 1)
        return pred


# ── Model Loader ──────────────────────────────────────────────
class TrafficPredictor:
    """
    Wraps the trained model with everything needed for inference.
    Loads once at startup, reused for every prediction request.
    """
    def __init__(self, checkpoint_path: str, adj_path: str,
                 scaler_path: str, config_path: str):

        with open(config_path, encoding='utf-8') as f:
            import json
            self.config = json.load(f)

        self.device     = torch.device("cpu")  # CPU for API serving
        self.n_sensors  = self.config["n_sensors"]
        self.n_features = self.config["n_features"]
        self.input_steps = self.config["input_steps"]
        self.pred_steps  = self.config["pred_steps"]
        self.feature_cols = self.config["feature_cols"]

        # Load scaler
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        self.speed_std  = float(self.scaler.scale_[0])
        self.speed_mean = float(self.scaler.mean_[0])

        # Load adjacency matrices
        adj_tensor = torch.load(adj_path, map_location="cpu")
        def norm_adj(a):
            r = a.sum(dim=1, keepdim=True).clamp(min=1e-8)
            return a / r
        self.adj_fwd = norm_adj(adj_tensor)
        self.adj_bwd = norm_adj(adj_tensor.T)

        # Inspect checkpoint to dynamically match architecture dimensions
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        ckpt_d_model = ckpt["model_state_dict"]["input_proj.weight"].shape[0]
        ckpt_n_heads = 8 if ckpt_d_model == 128 else 4

        # Build and load model
        self.model = SpatioTemporalTransformer(
            n_features=self.n_features, d_model=ckpt_d_model, n_heads=ckpt_n_heads,
            n_gnn_layers=2, n_tf_layers=2,
            n_sensors=self.n_sensors, pred_steps=self.pred_steps,
            dropout=0.0
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        print(f"✅ Model loaded (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}, d_model={ckpt_d_model})")

    def predict(self, history: np.ndarray) -> dict:
        """
        history: numpy array of shape (12, 325, 12)
                 — last 12 timesteps, all sensors, all features
        Returns: dict with predictions at 15, 30, 60 min in mph
        """
        x = torch.tensor(history[np.newaxis], dtype=torch.float32)
        with torch.no_grad():
            pred = self.model(x, self.adj_fwd, self.adj_bwd)
        pred_np = pred[0].numpy()   # (12, 325)

        # Convert scaled predictions back to mph
        speed_preds = pred_np * self.speed_std + self.speed_mean

        return {
            "15min": speed_preds[2].tolist(),   # step 3  = 15 min
            "30min": speed_preds[5].tolist(),   # step 6  = 30 min
            "60min": speed_preds[11].tolist(),  # step 12 = 60 min
        }
