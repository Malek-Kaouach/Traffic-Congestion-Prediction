import torch
import numpy as np

def normalize_adjacency_directed(adj):
    """Row-normalize directed graph adjacency matrix so rows sum to 1."""
    row_sum = adj.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return adj / row_sum

def inverse_scale_speed(y_scaled, scaler, speed_col_idx=0):
    """Convert normalized z-scores back to physical miles per hour (mph)."""
    mean = scaler.mean_[speed_col_idx]
    std = scaler.scale_[speed_col_idx]
    return y_scaled * std + mean

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mape(y_true, y_pred, eps=1e-5):
    mask = np.abs(y_true) > eps
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
