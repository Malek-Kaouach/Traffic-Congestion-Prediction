import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import random
import os
from datetime import datetime

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Congestion Predictor",
    page_icon="🚦",
    layout="wide"
)

# Allow environment variable override for Docker network communication (e.g., http://api:8000)
API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Sensor GPS coordinates (from PEMS-BAY metadata) ──────────
SENSOR_LOCATIONS = {
    0:   (37.2517, -121.9581),
    10:  (37.2507, -121.9125),
    20:  (37.2655, -121.9815),
    30:  (37.3031, -122.0345),
    40:  (37.3516, -122.0603),
    50:  (37.3837, -122.0681),
    60:  (37.4037, -122.0698),
    70:  (37.2627, -121.8585),
    80:  (37.2918, -121.8718),
    90:  (37.3204, -121.8903),
    100: (37.3626, -121.9175),
    110: (37.3746, -121.9311),
    120: (37.3912, -121.9955),
    130: (37.4021, -122.0416),
    140: (37.3297, -121.8421),
}

# ── Helper Functions ──────────────────────────────────────────
def speed_to_color(speed):
    if speed >= 60: return "#2ECC71"   # green  = low congestion
    if speed >= 40: return "#F39C12"   # orange = medium
    return "#E74C3C"                    # red    = high

def speed_to_congestion(speed):
    if speed >= 60: return "🟢 Low"
    if speed >= 40: return "🟡 Medium"
    return "🔴 High"

def make_sample_request(sensor_id, hour, dayofweek, weather):
    """Generate a realistic sample prediction request."""
    base_speed = 65 if hour not in range(7, 10) and hour not in range(16, 19) else 45
    history = []
    for t in range(12):
        history.append({
            "sensor_id":         sensor_id,
            "speed_mph":         base_speed + random.uniform(-5, 5),
            "hour":              hour,
            "dayofweek":         dayofweek,
            "temperature_f":     68.0,
            "humidity_pct":      65.0,
            "visibility_mi":     10.0,
            "wind_speed_mph":    5.0,
            "weather_condition": weather,
            "acc_count_60min":   0,
            "acc_max_severity":  0,
            "acc_mins_since":    60.0,
        })
    return {"history": history}


# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("🚦 Traffic Predictor")
st.sidebar.markdown("**STGTransformer v2**")
st.sidebar.markdown("GNN + Transformer | PEMS-BAY")
st.sidebar.divider()

sensor_id  = st.sidebar.selectbox("Sensor ID", list(SENSOR_LOCATIONS.keys()))
hour       = st.sidebar.slider("Hour of Day", 0, 23, 8)
dayofweek  = st.sidebar.selectbox("Day of Week",
    [0,1,2,3,4,5,6],
    format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x]
)
weather    = st.sidebar.selectbox("Weather Condition",
    ["Clear","Cloudy","Light Rain","Fog","Overcast"]
)
acc_count  = st.sidebar.slider("Nearby Accidents (past 60 min)", 0, 5, 0)
predict_btn = st.sidebar.button("🔮 Predict", type="primary", use_container_width=True)


# ── Main Content ──────────────────────────────────────────────
st.title("🚦 Real-Time Traffic Congestion Prediction")
st.markdown("**Spatiotemporal Transformer + GNN | SF Bay Area | PEMS-BAY Dataset**")
st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model", "STGTransformer v2")
with col2:
    st.metric("Sensors", "325")
with col3:
    st.metric("Area", "SF Bay Area")

st.divider()

# ── Prediction ────────────────────────────────────────────────
if predict_btn:
    with st.spinner("Running prediction..."):
        try:
            payload = make_sample_request(sensor_id, hour, dayofweek, weather)
            payload["history"][-1]["acc_count_60min"] = acc_count

            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=30)

            if resp.status_code == 200:
                result = resp.json()
                pred   = result["predictions"][0]
                summ   = result["summary"]

                st.success("✅ Prediction successful")

                # Metrics row
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric(
                        "15 min", f"{pred['speed_15min']} mph",
                        delta=f"{pred['congestion_level_15min']}"
                    )
                with c2:
                    st.metric(
                        "30 min", f"{pred['speed_30min']} mph",
                        delta=f"{pred['congestion_level_30min']}"
                    )
                with c3:
                    st.metric(
                        "60 min", f"{pred['speed_60min']} mph",
                        delta=f"{pred['congestion_level_60min']}"
                    )

                # Speed timeline chart
                st.subheader("Speed Forecast Timeline")
                times  = ["Now", "+15 min", "+30 min", "+45 min", "+60 min"]
                speeds = [
                    payload["history"][-1]["speed_mph"],
                    pred["speed_15min"],
                    (pred["speed_15min"] + pred["speed_30min"]) / 2,
                    pred["speed_30min"],
                    pred["speed_60min"],
                ]
                colors = [speed_to_color(s) for s in speeds]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=times, y=speeds,
                    mode="lines+markers",
                    line=dict(color="#378ADD", width=3),
                    marker=dict(size=12, color=colors),
                    name="Predicted Speed"
                ))
                fig.add_hline(y=60, line_dash="dash",
                              line_color="#2ECC71", annotation_text="Low congestion (60 mph)")
                fig.add_hline(y=40, line_dash="dash",
                              line_color="#E74C3C", annotation_text="High congestion (40 mph)")
                fig.update_layout(
                    yaxis_title="Speed (mph)",
                    yaxis_range=[0, 85],
                    height=350,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.error(f"API Error {resp.status_code}: {resp.text}")

        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to API at {API_URL}. Ensure the FastAPI server is running.")
        except Exception as e:
            st.error(f"Error: {e}")

else:
    st.info("👈 Configure prediction parameters in the sidebar and click **Predict**")


# ── Sensor Map ────────────────────────────────────────────────
st.subheader("📍 PEMS-BAY Sensor Network (Sample)")

map_data = []
for sid, (lat, lon) in SENSOR_LOCATIONS.items():
    map_data.append({
        "sensor_id": sid,
        "lat": lat,
        "lon": lon,
        "speed": random.uniform(40, 75),
    })

df_map = pd.DataFrame(map_data)
df_map["congestion"] = df_map["speed"].apply(
    lambda s: "Low" if s >= 60 else "Medium" if s >= 40 else "High"
)
df_map["color"] = df_map["speed"].apply(speed_to_color)

fig_map = px.scatter_mapbox(
    df_map, lat="lat", lon="lon",
    color="congestion",
    color_discrete_map={"Low":"#2ECC71","Medium":"#F39C12","High":"#E74C3C"},
    size_max=15,
    hover_data={"sensor_id": True, "speed": ":.1f", "lat": False, "lon": False},
    zoom=10,
    mapbox_style="open-street-map",
    title="Sensor Locations (color = congestion level)"
)
fig_map.update_traces(marker=dict(size=12))
fig_map.update_layout(height=450, margin=dict(l=0,r=0,t=30,b=0))
st.plotly_chart(fig_map, use_container_width=True)

# ── Model Info ────────────────────────────────────────────────
with st.expander("📊 Model Performance"):
    perf_data = {
        "Model":    ["Hist. Average","ARIMA","LSTM","STGTransformer (ours)",
                     "DCRNN (2018)","STGCN (2018)"],
        "15 min":   [2.93, 2.98, 1.47, 1.46, 1.38, 1.36],
        "30 min":   [3.11, 2.99, 1.97, 1.91, 1.74, 1.81],
        "60 min":   [3.54, 3.03, 2.61, 2.42, 2.07, 2.49],
    }
    df_perf = pd.DataFrame(perf_data)
    st.dataframe(
        df_perf.style.highlight_min(subset=["15 min","30 min","60 min"],
                                     color="#d4edda"),
        use_container_width=True
    )
    st.caption("MAE in mph. Lower is better. Green = best in column.")
