import numpy as np
import pickle
import json
from typing import Optional, List

# Complete weather categories matching training one-hot encoding
WEATHER_CATEGORIES = [
    'Clear', 'Fog', 'Haze', 'Heavy Rain', 'Light Drizzle',
    'Light Rain', 'Light Thunderstorms and Rain', 'Mist',
    'Mostly Cloudy', 'Overcast', 'Partly Cloudy', 'Patches of Fog',
    'Rain', 'Rain Showers', 'Scattered Clouds', 'Shallow Fog',
    'Squalls', 'Unknown'
]

WEATHER_MAP = {
    "Clear": 0, "Cloudy": 1, "Fog": 2, "Haze": 3,
    "Light Rain": 4, "Light Snow": 5, "Mostly Cloudy": 6,
    "Overcast": 7, "Rain": 8, "Snow": 9, "Unknown": 10
}


def build_feature_vector(
    speed: float,
    hour: int,
    dayofweek: int,
    temperature_f: float,
    humidity_pct: float,
    visibility_mi: float,
    wind_speed_mph: float,
    weather_condition: str,
    acc_count_60min: int,
    acc_max_severity: int,
    acc_mins_since: float,
    feature_cols: Optional[List[str]] = None
) -> np.ndarray:
    """
    Builds a single feature vector dynamically matching the model configuration.
    Supports both 31-feature (cyclical time + one-hot weather) and 12-feature formats.
    """
    is_weekend = 1.0 if dayofweek >= 5 else 0.0

    # If trained with 31 features (full multi-modal dataset)
    if feature_cols is not None and len(feature_cols) == 31:
        # 1. Cyclical time encodings
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        dow_sin  = np.sin(2 * np.pi * dayofweek / 7.0)
        dow_cos  = np.cos(2 * np.pi * dayofweek / 7.0)

        # Base continuous features
        vec = [
            speed, hour_sin, hour_cos, dow_sin, dow_cos, is_weekend,
            temperature_f, humidity_pct, visibility_mi, wind_speed_mph,
            float(acc_count_60min), float(acc_max_severity), float(acc_mins_since)
        ]

        # 2. One-hot weather encoding (18 categories)
        weather_clean = weather_condition.strip()
        for cat in WEATHER_CATEGORIES:
            vec.append(1.0 if weather_clean.lower() == cat.lower() else 0.0)

        return np.array(vec, dtype=np.float32)

    # Fallback: 12-feature format
    weather_code = WEATHER_MAP.get(weather_condition, WEATHER_MAP["Unknown"])
    return np.array([
        speed,
        float(hour),
        float(dayofweek),
        is_weekend,
        temperature_f,
        humidity_pct,
        visibility_mi,
        wind_speed_mph,
        float(weather_code),
        float(acc_count_60min),
        float(acc_max_severity),
        float(acc_mins_since),
    ], dtype=np.float32)


def normalize_features(
    feature_vector: np.ndarray,
    scaler
) -> np.ndarray:
    """Apply the same StandardScaler used during training."""
    return (feature_vector - scaler.mean_) / scaler.scale_
