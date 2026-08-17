from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import os
import sys

# Add deployment folder to path
sys.path.append(os.path.dirname(__file__))
from model import TrafficPredictor
from preprocess import build_feature_vector, normalize_features

# ── App Setup ─────────────────────────────────────────────────
app = FastAPI(
    title="Traffic Congestion Predictor",
    description="""
    Real-time traffic speed prediction using a Spatiotemporal
    Transformer + Graph Neural Network model trained on PEMS-BAY
    (325 sensors, San Francisco Bay Area).

    Multi-modal inputs: traffic speed history + weather + accidents.
    Predicts future speeds at 15, 30, and 60 minute horizons.
    """,
    version="1.0.0"
)

# Enable CORS for Streamlit / external frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Model Once at Startup ────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "model_files")

# Check for checkpoint with fallback
ckpt_name = "stgt_v2_best.pt" if os.path.exists(os.path.join(DATA_DIR, "stgt_v2_best.pt")) else "stgt_best.pt"

predictor = TrafficPredictor(
    checkpoint_path = os.path.join(DATA_DIR, ckpt_name),
    adj_path        = os.path.join(DATA_DIR, "adj_tensor.pt"),
    scaler_path     = os.path.join(DATA_DIR, "scaler.pkl"),
    config_path     = os.path.join(DATA_DIR, "config.json"),
)

# ── Request / Response Models ─────────────────────────────────
class SensorReading(BaseModel):
    sensor_id:         int   = Field(..., description="Sensor index (0-324)")
    speed_mph:         float = Field(..., description="Current speed in mph")
    hour:              int   = Field(..., description="Hour of day (0-23)")
    dayofweek:         int   = Field(..., description="Day of week (0=Mon, 6=Sun)")
    temperature_f:     float = Field(65.0, description="Temperature in Fahrenheit")
    humidity_pct:      float = Field(60.0, description="Humidity percentage")
    visibility_mi:     float = Field(10.0, description="Visibility in miles")
    wind_speed_mph:    float = Field(5.0,  description="Wind speed in mph")
    weather_condition: str   = Field("Clear", description="Weather condition string")
    acc_count_60min:   int   = Field(0,   description="Nearby accidents in past 60 min")
    acc_max_severity:  int   = Field(0,   description="Max severity (0-4)")
    acc_mins_since:    float = Field(60.0,description="Minutes since last accident")


class PredictionRequest(BaseModel):
    """Send the last 12 timesteps (1 hour) of sensor readings."""
    history: List[SensorReading] = Field(
        ..., min_items=12, max_items=12,
        description="Exactly 12 consecutive 5-min readings per sensor"
    )


class SpeedPrediction(BaseModel):
    sensor_id:  int
    speed_15min: float
    speed_30min: float
    speed_60min: float
    congestion_level_15min: str
    congestion_level_30min: str
    congestion_level_60min: str


class PredictionResponse(BaseModel):
    status:      str
    model:       str
    predictions: List[SpeedPrediction]
    summary:     dict


# ── Helper ────────────────────────────────────────────────────
def speed_to_congestion(speed_mph: float) -> str:
    if speed_mph >= 60:  return "Low"
    if speed_mph >= 40:  return "Medium"
    return "High"


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Traffic Congestion Predictor API",
        "model":   f"STGTransformer (GNN + Transformer, d_model={predictor.model.d_model})",
        "dataset": "PEMS-BAY (325 sensors, SF Bay Area)",
        "horizons": ["15 min", "30 min", "60 min"],
        "docs":    "/docs"
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Predict traffic speed for the next 15, 30, and 60 minutes.

    Send 12 consecutive sensor readings (1 hour of history).
    Each reading must include speed, time, weather, and accident context.
    Returns predicted speeds and congestion levels per sensor.
    """
    try:
        n_sensors   = predictor.n_sensors
        n_features  = predictor.n_features
        input_steps = predictor.input_steps

        # Build history array: (12, 325, n_features)
        history = np.zeros((input_steps, n_sensors, n_features), dtype=np.float32)

        for t, reading in enumerate(request.history):
            sid = reading.sensor_id
            if not (0 <= sid < n_sensors):
                raise HTTPException(
                    status_code=400,
                    detail=f"sensor_id {sid} out of range (0-{n_sensors-1})"
                )
            fv = build_feature_vector(
                speed             = reading.speed_mph,
                hour              = reading.hour,
                dayofweek         = reading.dayofweek,
                temperature_f     = reading.temperature_f,
                humidity_pct      = reading.humidity_pct,
                visibility_mi     = reading.visibility_mi,
                wind_speed_mph    = reading.wind_speed_mph,
                weather_condition = reading.weather_condition,
                acc_count_60min   = reading.acc_count_60min,
                acc_max_severity  = reading.acc_max_severity,
                acc_mins_since    = reading.acc_mins_since,
                feature_cols      = predictor.feature_cols
            )
            history[t, sid, :] = normalize_features(fv, predictor.scaler)

        # Run model
        preds = predictor.predict(history)

        # Build response
        sensor_ids = list(set(r.sensor_id for r in request.history))
        response_preds = []
        for sid in sensor_ids:
            s15 = round(preds["15min"][sid], 2)
            s30 = round(preds["30min"][sid], 2)
            s60 = round(preds["60min"][sid], 2)
            response_preds.append(SpeedPrediction(
                sensor_id              = sid,
                speed_15min            = s15,
                speed_30min            = s30,
                speed_60min            = s60,
                congestion_level_15min = speed_to_congestion(s15),
                congestion_level_30min = speed_to_congestion(s30),
                congestion_level_60min = speed_to_congestion(s60),
            ))

        avg_15 = round(np.mean(preds["15min"]), 2)
        avg_30 = round(np.mean(preds["30min"]), 2)
        avg_60 = round(np.mean(preds["60min"]), 2)

        return PredictionResponse(
            status      = "success",
            model       = f"STGTransformer (d_model={predictor.model.d_model})",
            predictions = response_preds,
            summary = {
                "avg_speed_15min_mph": avg_15,
                "avg_speed_30min_mph": avg_30,
                "avg_speed_60min_mph": avg_60,
                "avg_congestion_15min": speed_to_congestion(avg_15),
                "avg_congestion_30min": speed_to_congestion(avg_30),
                "avg_congestion_60min": speed_to_congestion(avg_60),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sensors/count")
def sensor_count():
    return {"n_sensors": predictor.n_sensors}


@app.get("/sensors/{sensor_id}/info")
def sensor_info(sensor_id: int):
    if not (0 <= sensor_id < predictor.n_sensors):
        raise HTTPException(status_code=404, detail="Sensor not found")
    return {
        "sensor_id": sensor_id,
        "location":  "SF Bay Area highway network",
        "sampling":  "Every 5 minutes",
        "features":  predictor.feature_cols,
    }
