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
# FastAPI base (inside docker-compose the hostname is the service name "fastapi")
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", 8800))
API_BASE = f"http://fastapi:{FASTAPI_PORT}"

# Endpoints used by the UI
PREDICT_BY_ID_ENDPOINT = f"{API_BASE}/predict/by_id"
FEATURE_IDS_ENDPOINT   = f"{API_BASE}/features/ids"
HEALTH_ENDPOINT        = f"{API_BASE}/health"

# Default BA center (roughly Obelisco) for initial map state
DEFAULT_CENTER = {"lat": -34.6037, "lon": -58.3816}

st.set_page_config(
    page_title="Airbnb Occupancy Predictor (Buenos Aires)",
    page_icon="🏠",
    layout="centered",
)

# =========================
# ----- HELPERS -----------
# =========================
@st.cache_data(ttl=300)
def get_available_ids() -> list[int]:
    """
    Ask the API for the list of available listing_ids (preprocessed features).
    The API should return: {"ids": [int, int, ...]}
    If the endpoint is not available or fails, return an empty list.
    """
    try:
        r = requests.get(FEATURE_IDS_ENDPOINT, timeout=10)
        if r.status_code == 200:
            data = r.json()
            ids = data.get("ids", [])
            return [int(x) for x in ids if str(x).isdigit()]
    except Exception:
        pass
    return []

def _badge(text: str, color: str):
    """Render a small colored label/badge."""
    st.markdown(
        f"""
        <span style="background-color:{color};color:white;padding:4px 10px;border-radius:12px;font-size:0.9rem;">
        {text}
        </span>
        """,
        unsafe_allow_html=True,
    )

def render_free_map(lat: float, lon: float, listing_id_out, score, ui_pred):
    """
    Render a map using FREE tiles (Carto light_all) via a TileLayer.
    - No Mapbox token required.
    - Shows the selected listing as a colored point.
    - If coords look normalized/invalid, center on Buenos Aires with a warning.
    """
    # Basic sanity for "normalized" coords (0.x / -0.x). Adjust as needed for your data.
    try:
        lat_f, lon_f = float(lat), float(lon)
    except Exception:
        lat_f, lon_f = DEFAULT_CENTER["lat"], DEFAULT_CENTER["lon"]

    invalidish = (-1.0 <= lat_f <= 1.0) and (-1.0 <= lon_f <= 1.0)
    if invalidish:
        st.warning("Coordinates look normalized (not real latitude/longitude). Centering map on Buenos Aires.")
        center_lat, center_lon = DEFAULT_CENTER["lat"], DEFAULT_CENTER["lon"]
    else:
        center_lat, center_lon = lat_f, lon_f

    # Point color based on prediction
    color = [22, 163, 74] if ui_pred == 1 else [220, 38, 38]  # green/red

    # Scatter point layer
    point_layer = pdk.Layer(
        "ScatterplotLayer",
        data=[{"lat": lat_f, "lon": lon_f}],
        get_position=["lon", "lat"],
        get_radius=80,
        pickable=True,
        filled=True,
        get_fill_color=color,
    )

    # FREE basemap (Carto)
    tile_layer = pdk.Layer(
        "TileLayer",
        data="https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        min_zoom=0,
        max_zoom=19,
        tile_size=256,
    )

    # View centered on listing (or BA if invalid)
    view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=12, pitch=0)

    tooltip_text = f"listing_id: {listing_id_out}"
    if score is not None:
        try:
            tooltip_text += f"\nscore: {float(score):.3f}"
        except Exception:
            pass

    deck = pdk.Deck(
        layers=[tile_layer, point_layer],
        initial_view_state=view_state,
        tooltip={"text": tooltip_text},
        map_style=None,  # IMPORTANT when using a custom TileLayer
    )
    st.pydeck_chart(deck, use_container_width=True)

# =========================
# ----- SIDEBAR -----------
# =========================
st.sidebar.title("Settings")

with st.sidebar.expander("API Settings", expanded=False):
    # Just show where we are pointing to
    st.text_input("API base URL", value=API_BASE, key="api_base_display", disabled=True)

with st.sidebar.expander("Classification Threshold", expanded=True):
    # Let users explore different thresholds over the returned score
    threshold = st.slider("Threshold (score ≥ threshold → High Occupancy)", 0.0, 1.0, 0.5, 0.01)

with st.sidebar.expander("Available listing_id", expanded=False):
    """
    Dropdown fed by /features/ids. Selecting here fills the text box in the main form.
    """
    ids = get_available_ids()
    if ids:
        selected_id = st.selectbox("Pick one", ids, key="select_listing_id")
        if st.button("Use selected ID"):
            st.session_state["listing_id_input"] = str(selected_id)
            st.success(f"Selected listing_id {selected_id} copied to the form.")
    else:
        st.info("No IDs available from the API (or endpoint /features/ids not implemented).")

# =========================
# ----- HEADER ------------
# =========================
st.title("🏠 Airbnb Occupancy Predictor — Buenos Aires")
st.caption("Look up a preprocessed listing by its `listing_id`, visualize it on the map, and see the predicted occupancy.")

# Soft health check (non-blocking)
try:
    _ = requests.get(HEALTH_ENDPOINT, timeout=2)
except Exception:
    st.warning("API not reachable right now. Ensure the FastAPI service is up.")

# =========================
# ----- SEARCH FORM -------
# =========================
with st.form(key="lookup_form"):
    # If the user clicked "Use selected ID" in the sidebar, preload it here
    default_text = st.session_state.get("listing_id_input", "")
    listing_id = st.text_input(
        "Enter listing_id",
        value=default_text,
        help="This ID must exist in the preprocessed features."
    )
    submitted = st.form_submit_button("Predict")

# =========================
# ----- PREDICT FLOW ------
# =========================
if submitted:
    # Basic validation
    if not listing_id.strip().isdigit():
        st.error("`listing_id` must be an integer.")
        st.stop()

    with st.spinner("Fetching listing and scoring..."):
        # Call /predict/by_id (which reads preprocessed features from S3/MinIO on the backend)
        try:
            payload = {"ids": [int(listing_id)]}  # Backend expects {"ids": [...]}
            resp = requests.post(
                PREDICT_BY_ID_ENDPOINT,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
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

    # Expected API shape:
    # {
    #   "count": <int>,
    #   "missing_ids": [ ... ],
    #   "results": [
    #       {"listing_id": ..., "prediction": 0/1, "score": <float|None>, "lat": <float|None>,
    #        "lon": <float|None>, "model_version": <int>}
    #   ]
    # }
    missing = data.get("missing_ids", [])
    if missing:
        st.error(f"listing_id not found: {missing}")
        st.stop()

    results = data.get("results", [])
    if not results:
        st.error("The API returned no results.")
        st.stop()

    row = results[0]
    pred = int(row.get("prediction", 0))
    score = row.get("score", None)
    lat = row.get("lat", None)
    lon = row.get("lon", None)
    model_version = row.get("model_version", None)
    listing_id_out = row.get("listing_id", listing_id)

    # Compute the UI label using the interactive threshold if we received a probability
    ui_pred = int(float(score) >= threshold) if score is not None else pred

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
            st.metric("Score (probability)", f"{float(score):.3f}",
                      help="Probability of high occupancy (class=1).")
        st.caption(f"Threshold: **{threshold:.2f}**  •  Model version: **{model_version}**")

    with col2:
        st.write("**Listing**")
        st.write(f"`listing_id`: **{listing_id_out}**")
        if (lat is not None) and (lon is not None):
            st.write(f"Lat/Lon: **{float(lat):.5f}**, **{float(lon):.5f}**")
        else:
            st.write("_No coordinates available for this listing._")

    # =========================
    # ----- MAP VIEW ----------
    # =========================
    st.subheader("Map")
    if (lat is not None) and (lon is not None):
        render_free_map(lat, lon, listing_id_out, score, ui_pred)
    else:
        st.info("No coordinates to display. Showing default map center.")
        st.map(pd.DataFrame([DEFAULT_CENTER]), zoom=10)

    # =========================
    # ----- RAW RESPONSE ------
    # =========================
    with st.expander("Raw API response", expanded=False):
        st.json(data)

st.markdown("---")
st.caption(
    "Abril Noguera - José Roberto Castro - Kevin Nelson Pennington - "
    "Pablo Ezequiel Brahim | Experiment: Airbnb Buenos Aires • Data from Inside Airbnb"
)