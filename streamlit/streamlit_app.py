# streamlit_app.py
import os
import json
import requests
import pandas as pd
import streamlit as st
import pydeck as pdk

# =========================
# ----- CONFIG ------------
# =========================
# Point this to your FastAPI service (e.g., http://fastapi:8800 in docker-compose, or http://localhost:8800 locally)
FASTAPI_PORT = os.getenv("FASTAPI_PORT", 8800)
API_BASE = f"http://fastapi:{FASTAPI_PORT}"

# Endpoints
PREDICT_BY_ID_ENDPOINT = f"{API_BASE}/predict/by-id"
HEALTH_ENDPOINT = f"{API_BASE}/"

# Default BA center (roughly obelisco) for initial map state
DEFAULT_CENTER = {"lat": -34.6037, "lon": -58.3816}

st.set_page_config(
    page_title="Airbnb Occupancy Predictor (Buenos Aires)",
    page_icon="🏠",
    layout="centered",
)

# =========================
# ----- SIDEBAR -----------
# =========================
st.sidebar.title("Settings")
with st.sidebar.expander("API Settings", expanded=False):
    st.text_input("API base URL", value=API_BASE, key="api_base_display", disabled=True)
with st.sidebar.expander("Classification Threshold", expanded=True):
    # Let users explore different thresholds on the returned score
    threshold = st.slider("Threshold (score ≥ threshold → High Occupancy)", 0.0, 1.0, 0.5, 0.01)

# =========================
# ----- HEADER ------------
# =========================
st.title("🏠 Airbnb Occupancy Predictor — Buenos Aires")
st.caption("Look up a preprocessed listing by its `listing_id`, visualize it on the map, and see the predicted occupancy.")

# Health check (non-blocking soft fail)
try:
    _ = requests.get(HEALTH_ENDPOINT, timeout=2)
except Exception:
    st.warning("API not reachable right now. Ensure the FastAPI service is up.")

# =========================
# ----- SEARCH FORM -------
# =========================
with st.form(key="lookup_form"):
    listing_id = st.text_input("Enter listing_id", value="", help="This ID must exist in the preprocessed train/test features.")
    submitted = st.form_submit_button("Predict")

def _badge(text, color):
    """Small colored label/badge."""
    st.markdown(
        f"""
        <span style="background-color:{color};color:white;padding:4px 10px;border-radius:12px;font-size:0.9rem;">
        {text}
        </span>
        """,
        unsafe_allow_html=True,
    )

# =========================
# ----- PREDICT FLOW ------
# =========================
if submitted:
    if not listing_id.strip().isdigit():
        st.error("`listing_id` must be an integer.")
        st.stop()

    with st.spinner("Fetching listing and scoring..."):
        try:
            # Call the recommended /predict/by-id path (uses preprocessed features from S3)
            resp = requests.post(
                PREDICT_BY_ID_ENDPOINT,
                headers={"Content-Type": "application/json"},
                data=json.dumps({"listing_id": int(listing_id)}),
                timeout=30,
            )
        except Exception as e:
            st.error(f"Request error: {e}")
            st.stop()

    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        st.error(f"API error ({resp.status_code}): {detail}")
        st.stop()

    data = resp.json()
    # Expected fields from API: listing_id, prediction, score, lat, lon, model_version
    pred = int(data.get("prediction", 0))
    score = data.get("score", None)
    lat = data.get("lat", None)
    lon = data.get("lon", None)
    model_version = data.get("model_version", None)

    # Derive predicted label using current UI threshold (instead of the fixed 0.5 in backend)
    # If score is None (model without predict_proba), fall back to API's binary prediction
    if score is not None:
        ui_pred = int(float(score) >= threshold)
    else:
        ui_pred = pred

    # =========================
    # ----- RESULT CARD -------
    # =========================
    st.subheader("Prediction")
    col1, col2 = st.columns(2)
    with col1:
        if ui_pred == 1:
            _badge("High Occupancy", "#16a34a")  # green
        else:
            _badge("Low Occupancy", "#dc2626")   # red

        if score is not None:
            st.metric("Score (probability)", f"{float(score):.3f}", help="Probability of high occupancy (class=1).")
        st.caption(f"Threshold: **{threshold:.2f}**  •  Model version: **{model_version}**")

    with col2:
        st.write("**Listing**")
        st.write(f"`listing_id`: **{data.get('listing_id')}**")
        if lat is not None and lon is not None:
            st.write(f"Lat/Lon: **{lat:.5f}**, **{lon:.5f}**")
        else:
            st.write("_No coordinates available for this listing._")

    # =========================
    # ----- MAP VIEW ----------
    # =========================
    st.subheader("Map")
    if lat is not None and lon is not None:
        map_df = pd.DataFrame([{"lat": lat, "lon": lon}])
        tooltip = {"text": f"listing_id: {data.get('listing_id')}\nscore: {score:.3f}" if score is not None else f"listing_id: {data.get('listing_id')}"}
        color = [22, 163, 74] if ui_pred == 1 else [220, 38, 38]  # green/red RGB

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position=["lon", "lat"],
            get_radius=60,
            get_fill_color=color,
            pickable=True,
        )

        view_state = pdk.ViewState(
            latitude=float(lat),
            longitude=float(lon),
            zoom=12,
            pitch=0,
        )

        r = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style="mapbox://styles/mapbox/light-v9"  # Streamlit will handle token; if not, plain basemap.
        )
        st.pydeck_chart(r, use_container_width=True)
    else:
        st.info("No coordinates to display. Showing default map center.")
        st.map(pd.DataFrame([DEFAULT_CENTER]), zoom=10)

    # =========================
    # ----- RAW RESPONSE ------
    # =========================
    with st.expander("Raw API response", expanded=False):
        st.json(data)

st.markdown("---")
st.caption("Abril Noguera - José Roberto Castro - Kevin Nelson Pennington - Pablo Ezequiel Brahim | Experiment: Airbnb Buenos Aires • Data from Inside Airbnb")