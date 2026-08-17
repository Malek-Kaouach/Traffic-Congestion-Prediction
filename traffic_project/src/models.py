import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionConvLayer(nn.Module):
    """Bidirectional Diffusion Graph Convolution."""
    def __init__(self, d_model, n_hops=2):
        super().__init__()
        self.n_hops = n_hops
        self.fwd_linears = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(n_hops)
        ])
        self.bwd_linears = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(n_hops)
        ])
        self.out = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, adj_fwd, adj_bwd):
        h_fwd = x
        for layer in self.fwd_linears:
            h_fwd = F.relu(layer(torch.einsum('nm, bmd -> bnd', adj_fwd, h_fwd)))

        h_bwd = x
        for layer in self.bwd_linears:
            h_bwd = F.relu(layer(torch.einsum('nm, bmd -> bnd', adj_bwd, h_bwd)))

        combined = torch.cat([h_fwd, h_bwd], dim=-1)
        out = self.out(combined)
        return self.norm(out + x)


class SpatioTemporalTransformer(nn.Module):
    """Spatial-Temporal Transformer architecture combining GNN and Transformer Encoder."""
    def __init__(self, n_features, d_model, n_heads, n_gnn_layers, n_tf_layers, n_sensors, pred_steps, input_steps=12, dropout=0.1):
        super().__init__()
        self.n_sensors   = n_sensors
        self.pred_steps  = pred_steps
        self.input_steps = input_steps
        self.d_model     = d_model

        self.input_proj = nn.Linear(n_features, d_model)
        self.gnn_layers = nn.ModuleList([
            DiffusionConvLayer(d_model, n_hops=2) for _ in range(n_gnn_layers)
        ])
        self.spatial_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, input_steps, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_tf_layers)
        self.tf_norm = nn.LayerNorm(d_model)

        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, pred_steps)
        )

    def forward(self, x, adj_fwd, adj_bwd):
        B, T, N, n_feat = x.shape

        x = self.input_proj(x)
        x_orig = x.clone()

        x_flat = x.reshape(B * T, N, self.d_model)
        for gnn in self.gnn_layers:
            x_flat = gnn(x_flat, adj_fwd, adj_bwd)

        x_spatial = x_flat.reshape(B, T, N, self.d_model)

        gate_input = torch.cat([x_spatial, x_orig], dim=-1)
        gate       = self.spatial_gate(gate_input)
        x          = gate * x_spatial + (1 - gate) * x_orig

        x = x.permute(0, 2, 1, 3).reshape(B * N, T, self.d_model)
        x = x + self.pos_embedding[:, :T, :]
        x = self.transformer(x)
        x = self.tf_norm(x[:, -1, :])

        pred = self.pred_head(x)
        pred = pred.reshape(B, N, self.pred_steps)
        pred = pred.permute(0, 2, 1)
        return pred
