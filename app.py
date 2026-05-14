import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk
from datetime import datetime
from inference import predict_load


# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="PJME Load Forecasting",
    page_icon="⚡",
    layout="centered"
)


# ============================================================
# Compact CSS
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 900px;
        padding-top: 0.8rem;
        padding-bottom: 0.8rem;
    }

    h1 {
        font-size: 1.6rem !important;
        margin-bottom: 0.25rem !important;
    }

    h2, h3 {
        font-size: 1.15rem !important;
        margin-top: 0.45rem !important;
        margin-bottom: 0.25rem !important;
    }

    p {
        font-size: 0.9rem !important;
        margin-bottom: 0.35rem !important;
    }

    .stCaption {
        font-size: 0.78rem !important;
    }

    .stSlider {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        margin-top: -0.25rem !important;
        margin-bottom: -0.35rem !important;
    }

    .stSlider label {
        font-size: 0.78rem !important;
        margin-bottom: -0.3rem !important;
    }

    div[data-testid="stSlider"] {
        min-height: 46px !important;
    }

    div[data-testid="stMetric"] {
        background-color: #f7f7f7;
        padding: 0.75rem;
        border-radius: 12px;
        border: 1px solid #e5e5e5;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        font-weight: 700 !important;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.55rem !important;
        font-weight: 800 !important;
    }

    div[data-testid="stMetricDelta"] {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stNumberInput"] label {
        font-size: 0.78rem !important;
    }

    div[data-testid="stSelectbox"] {
        margin-bottom: -0.2rem !important;
    }

    div[data-testid="stNumberInput"] {
        margin-bottom: -0.2rem !important;
    }

    .small-caption {
        font-size: 0.75rem;
        color: #666666;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Header
# ============================================================

st.title("Interactive PJME Load Forecasting Dashboard")

st.write(
    "Adjust regional electricity loads and select the prediction time. "
    "The PJME load prediction updates automatically, and the map colour changes according to load intensity."
)

with st.expander("Model Information", expanded=False):
    st.markdown(
        """
        **Model type:** Multi-region LSTM  
        **Prediction target:** PJME load  
        **Input feature setting:** 21-feature version  

        The app uses:
        - `final_model.pth`
        - `scaler.pkl`
        - `feature_cols.pkl`
        """
    )


# ============================================================
# Region Coordinates
# ============================================================

region_info = {
    "AEP": {"lat": 39.9612, "lon": -82.9988, "default": 15000.0},
    "COMED": {"lat": 41.8781, "lon": -87.6298, "default": 12000.0},
    "DAYTON": {"lat": 39.7589, "lon": -84.1916, "default": 3000.0},
    "DEOK": {"lat": 39.1031, "lon": -84.5120, "default": 4000.0},
    "DOM": {"lat": 37.5407, "lon": -77.4360, "default": 5000.0},
    "DUQ": {"lat": 40.4406, "lon": -79.9959, "default": 2000.0},
    "EKPC": {"lat": 38.2009, "lon": -84.8733, "default": 2500.0},
    "FE": {"lat": 41.0814, "lon": -81.5190, "default": 6000.0},
    "NI": {"lat": 41.5868, "lon": -87.3464, "default": 11000.0},
    "PJME": {"lat": 39.9526, "lon": -75.1652, "default": 10000.0},
    "PJMW": {"lat": 40.7128, "lon": -74.0060, "default": 7000.0},
}


# ============================================================
# 1. Current PJME Input
# ============================================================

st.subheader("1. Current PJME Load")

st.caption(
    "This is the current observed PJME load. The model uses this value together with other regional loads "
    "and time features to forecast the next PJME load."
)

pjme = st.slider(
    "Current PJME Load",
    min_value=0.0,
    max_value=30000.0,
    value=region_info["PJME"]["default"],
    step=100.0
)


# ============================================================
# 2. Other Regional Load Inputs
# ============================================================

st.subheader("2. Other Regional Load Inputs")

st.caption(
    "Move the sliders to simulate different regional electricity load conditions. "
    "The prediction and map update automatically."
)

load_col1, load_col2, load_col3, load_col4 = st.columns(4)

with load_col1:
    aep = st.slider("AEP", 0.0, 30000.0, region_info["AEP"]["default"], 100.0)
    comed = st.slider("COMED", 0.0, 30000.0, region_info["COMED"]["default"], 100.0)
    dayton = st.slider("DAYTON", 0.0, 30000.0, region_info["DAYTON"]["default"], 100.0)

with load_col2:
    deok = st.slider("DEOK", 0.0, 30000.0, region_info["DEOK"]["default"], 100.0)
    dom = st.slider("DOM", 0.0, 30000.0, region_info["DOM"]["default"], 100.0)
    duq = st.slider("DUQ", 0.0, 30000.0, region_info["DUQ"]["default"], 100.0)

with load_col3:
    ekpc = st.slider("EKPC", 0.0, 30000.0, region_info["EKPC"]["default"], 100.0)
    fe = st.slider("FE", 0.0, 30000.0, region_info["FE"]["default"], 100.0)

with load_col4:
    ni = st.slider("NI", 0.0, 30000.0, region_info["NI"]["default"], 100.0)
    pjmw = st.slider("PJMW", 0.0, 30000.0, region_info["PJMW"]["default"], 100.0)


regional_loads = {
    "AEP": aep,
    "COMED": comed,
    "DAYTON": dayton,
    "DEOK": deok,
    "DOM": dom,
    "DUQ": duq,
    "EKPC": ekpc,
    "FE": fe,
    "NI": ni,
    "PJME": pjme,
    "PJMW": pjmw,
}


# ============================================================
# 3. Time Selection
# ============================================================

st.subheader("3. Time Selection")

st.caption(
    "Select the date and hour for prediction. "
    "The system automatically calculates day of week and day of year for the trained model."
)

time_col1, time_col2, time_col3, time_col4 = st.columns(4)

hour_options = {
    "00:00": 0,
    "01:00": 1,
    "02:00": 2,
    "03:00": 3,
    "04:00": 4,
    "05:00": 5,
    "06:00": 6,
    "07:00": 7,
    "08:00": 8,
    "09:00": 9,
    "10:00": 10,
    "11:00": 11,
    "12:00": 12,
    "13:00": 13,
    "14:00": 14,
    "15:00": 15,
    "16:00": 16,
    "17:00": 17,
    "18:00": 18,
    "19:00": 19,
    "20:00": 20,
    "21:00": 21,
    "22:00": 22,
    "23:00": 23,
}

month_options = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


def is_leap_year(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def get_max_day(year, month):
    if month == 2:
        return 29 if is_leap_year(year) else 28
    if month in [4, 6, 9, 11]:
        return 30
    return 31


with time_col1:
    year_options = list(range(2010, 2036))

    selected_year = st.selectbox(
        "Year",
        options=year_options,
        index=year_options.index(2024)
    )

with time_col2:
    selected_month = st.selectbox(
        "Month",
        options=list(month_options.keys()),
        index=7
    )
    month = month_options[selected_month]

max_day = get_max_day(selected_year, month)

with time_col3:
    day_of_month = st.number_input(
        "Day",
        min_value=1,
        max_value=max_day,
        value=min(15, max_day),
        step=1
    )

with time_col4:
    selected_hour = st.selectbox(
        "Hour of Day",
        options=list(hour_options.keys()),
        index=18
    )
    hour = hour_options[selected_hour]


selected_date = datetime(selected_year, month, int(day_of_month))

day_of_week = selected_date.weekday()
day_of_year = selected_date.timetuple().tm_yday


# ============================================================
# Feature Engineering
# ============================================================

hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)

day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7)
day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7)

month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)

user_inputs = {
    "AEP": aep,
    "COMED": comed,
    "DAYTON": dayton,
    "DEOK": deok,
    "DOM": dom,
    "DUQ": duq,
    "EKPC": ekpc,
    "FE": fe,
    "NI": ni,
    "PJME": pjme,
    "PJMW": pjmw,

    "hour": hour,
    "day_of_week": day_of_week,
    "month": month,
    "day_of_year": day_of_year,

    "hour_sin": hour_sin,
    "hour_cos": hour_cos,
    "day_of_week_sin": day_of_week_sin,
    "day_of_week_cos": day_of_week_cos,
    "month_sin": month_sin,
    "month_cos": month_cos,
}


# ============================================================
# 4. Real-time Prediction
# ============================================================

st.subheader("4. Real-time PJME Prediction")

try:
    prediction = predict_load(user_inputs)

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric(
            label="Current PJME Input",
            value=f"{pjme:,.2f} MW"
        )

    with metric_col2:
        st.metric(
            label="Predicted PJME Load",
            value=f"{prediction:,.2f} MW"
        )

    with metric_col3:
        st.metric(
            label="Predicted Change",
            value=f"{prediction - pjme:,.2f} MW"
        )

except Exception as e:
    prediction = None
    st.error(f"Prediction failed: {e}")


# ============================================================
# Map Data Preparation
# ============================================================

def get_color(load_value, min_load=0, max_load=30000):
    """
    Convert load value into RGB colour.

    Low load  -> green
    High load -> red
    """
    ratio = (load_value - min_load) / (max_load - min_load)
    ratio = max(0, min(1, ratio))

    red = int(255 * ratio)
    green = int(255 * (1 - ratio))
    blue = 60

    return [red, green, blue, 180]


map_rows = []

for region, info in region_info.items():
    load_value = regional_loads[region]
    is_target = region == "PJME"

    map_rows.append(
        {
            "region": region,
            "lat": info["lat"],
            "lon": info["lon"],
            "load": load_value,
            "is_target": is_target,
            "label": "PJME Target" if is_target else region,

            # PJME is larger because it is the prediction target region
            "radius": 30000 + load_value * 2.2 if is_target else 15000 + load_value * 1.8,

            # Load-based colour
            "color": get_color(load_value),

            # PJME has stronger border
            "line_color": [0, 0, 0, 255] if is_target else [80, 80, 80, 120],
            "line_width": 5 if is_target else 1,
        }
    )

map_df = pd.DataFrame(map_rows)

map_df_other = map_df[map_df["region"] != "PJME"]
map_df_pjme = map_df[map_df["region"] == "PJME"]


# ============================================================
# 5. Regional Load Intensity Map
# ============================================================

st.subheader("5. Regional Load Intensity Map")

other_region_layer = pdk.Layer(
    "ScatterplotLayer",
    data=map_df_other,
    get_position="[lon, lat]",
    get_radius="radius",
    get_fill_color="color",
    get_line_color="line_color",
    get_line_width="line_width",
    stroked=True,
    filled=True,
    pickable=True,
    auto_highlight=True,
)

pjme_target_layer = pdk.Layer(
    "ScatterplotLayer",
    data=map_df_pjme,
    get_position="[lon, lat]",
    get_radius="radius",
    get_fill_color="color",
    get_line_color="line_color",
    get_line_width="line_width",
    stroked=True,
    filled=True,
    pickable=True,
    auto_highlight=True,
)

pjme_label_layer = pdk.Layer(
    "TextLayer",
    data=map_df_pjme,
    get_position="[lon, lat]",
    get_text="label",
    get_size=18,
    get_color=[0, 0, 0, 255],
    get_angle=0,
    get_text_anchor='"middle"',
    get_alignment_baseline='"bottom"',
    pickable=False,
)

view_state = pdk.ViewState(
    latitude=39.5,
    longitude=-82.0,
    zoom=4.0,
    pitch=25,
)

tooltip = {
    "html": """
    <b>Region:</b> {region}<br/>
    <b>Load:</b> {load} MW<br/>
    <b>Role:</b> {label}
    """,
    "style": {
        "backgroundColor": "steelblue",
        "color": "white",
        "fontSize": "13px",
        "padding": "8px",
    },
}

deck = pdk.Deck(
    layers=[
        other_region_layer,
        pjme_target_layer,
        pjme_label_layer,
    ],
    initial_view_state=view_state,
    tooltip=tooltip,
    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
)

st.pydeck_chart(deck, use_container_width=True, height=360)

st.caption(
    "The map uses heat-style circles to represent regional load intensity. "
    "Green means lower load, while red means higher load. "
    "PJME is highlighted with a larger circle, thicker border, and label because it is the prediction target region."
)


# ============================================================
# Optional Details
# ============================================================

with st.expander("Show input values used for prediction", expanded=False):
    st.json(user_inputs)

with st.expander("Show regional load table", expanded=False):
    st.dataframe(
        map_df[["region", "load", "lat", "lon", "label"]],
        use_container_width=True
    )

st.caption(
    "This compact interactive dashboard demonstrates a 21-feature LSTM-based PJME load forecasting pipeline."
)