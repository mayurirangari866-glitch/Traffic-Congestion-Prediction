
import streamlit as st
import pandas as pd
import joblib
import altair as alt
import requests
import os

from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_FOLDER = Path(__file__).resolve().parent

load_dotenv(PROJECT_FOLDER / ".env")

# Read the API key from Streamlit Cloud Secrets first, then local .env.
try:
    TOMTOM_API_KEY = st.secrets.get("TOMTOM_API_KEY", "")
except Exception:
    TOMTOM_API_KEY = ""

if not TOMTOM_API_KEY:
    TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY", "")

TOMTOM_API_KEY = TOMTOM_API_KEY.strip() if TOMTOM_API_KEY else ""

# Reject common placeholder values so they are never sent to TomTom.
if TOMTOM_API_KEY.lower() in {
    "your_actual_tomtom_api_key",
    "your_tomtom_api_key_here",
    "your_api_key",
    ""
}:
    TOMTOM_API_KEY = ""

MODEL_PATH = PROJECT_FOLDER / "models" / "traffic_volume_model.pkl"
DATA_PATH = PROJECT_FOLDER / "outputs" / "cleaned_traffic.csv"
COMPARISON_PATH = PROJECT_FOLDER / "outputs" / "model_comparison.csv"
IMPORTANCE_PATH = PROJECT_FOLDER / "outputs" / "feature_importance.png"

LIVE_TRAFFIC_PATH = PROJECT_FOLDER / "outputs" / "live_traffic_data.csv"
LIVE_WEATHER_PATH = PROJECT_FOLDER / "outputs" / "live_weather_data.csv"
COMBINED_LIVE_PATH = PROJECT_FOLDER / "outputs" / "combined_live_data.csv"


# --------------------------------------------------
# LIVE API DATA COLLECTION
# --------------------------------------------------

def collect_live_api_data():
    """Collect traffic from two Nagpur points and weather data."""

    if not TOMTOM_API_KEY:
        st.error("TomTom API key is missing or still set to a placeholder. Add your real key in Streamlit Cloud: App settings → Secrets, under TOMTOM_API_KEY.")
        return

    output_folder = PROJECT_FOLDER / "outputs"
    output_folder.mkdir(exist_ok=True)

    traffic_path = output_folder / "live_traffic_multi_location.csv"
    weather_path = output_folder / "live_weather_data.csv"

    locations = [
        {
            "location_name": "Existing Nagpur Monitoring Point",
            "latitude": 21.1458,
            "longitude": 79.0882
        },
        {
            "location_name": "Chhatrapati Square - Wardha Road",
            "latitude": 21.11083,
            "longitude": 79.07011
        }
    ]

    # Collect traffic from both locations
    for location in locations:

        try:
            traffic_url = (
                "https://api.tomtom.com/traffic/services/4/"
                "flowSegmentData/relative/10/json"
            )

            traffic_params = {
                "point": (
                    f"{location['latitude']},"
                    f"{location['longitude']}"
                ),
                "unit": "KMPH",
                "key": TOMTOM_API_KEY
            }

            response = requests.get(
                traffic_url,
                params=traffic_params,
                timeout=20
            )

            response.raise_for_status()

            traffic_data = response.json()["flowSegmentData"]

            traffic_row = {
                "timestamp_utc": pd.Timestamp.now(
                    tz="UTC"
                ).isoformat(),

                "location_name": location["location_name"],
                "latitude": location["latitude"],
                "longitude": location["longitude"],

                "current_speed_kmph": traffic_data.get("currentSpeed"),
                "free_flow_speed_kmph": traffic_data.get("freeFlowSpeed"),
                "confidence": traffic_data.get("confidence"),
                "road_closure": traffic_data.get("roadClosure")
            }

            pd.DataFrame([traffic_row]).to_csv(
                traffic_path,
                mode="a",
                header=not traffic_path.exists(),
                index=False
            )

            st.success(
                f"Traffic updated: {location['location_name']}"
            )

        except requests.HTTPError as error:
            # Do not print the request URL: it contains the API key.
            status = error.response.status_code if error.response is not None else "unknown"
            if status == 401 or status == 403:
                st.warning(
                    f"TomTom rejected the API key for {location['location_name']} "
                    f"(HTTP {status}). Check the TOMTOM_API_KEY value in Streamlit Cloud Secrets."
                )
            else:
                st.warning(
                    f"TomTom traffic request failed for {location['location_name']} "
                    f"(HTTP {status}). Please try again later."
                )
        except requests.RequestException:
            st.warning(
                f"Could not connect to TomTom for {location['location_name']}. "
                "Please check the network connection and try again."
            )
        except (KeyError, ValueError):
            st.warning(
                f"TomTom returned an unexpected response for {location['location_name']}."
            )

    # Collect weather once for the monitoring area
    try:
        weather_url = "https://api.open-meteo.com/v1/forecast"

        weather_params = {
            "latitude": 21.1458,
            "longitude": 79.0882,

            "current": (
                "temperature_2m,relative_humidity_2m,"
                "precipitation,rain,weather_code,wind_speed_10m"
            ),

            "timezone": "Asia/Kolkata"
        }

        response = requests.get(
            weather_url,
            params=weather_params,
            timeout=20
        )

        response.raise_for_status()

        weather_data = response.json()["current"]

        weather_row = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "weather_time": weather_data.get("time"),

            "latitude": 21.1458,
            "longitude": 79.0882,

            "temperature_c": weather_data.get("temperature_2m"),
            "humidity_percent": weather_data.get("relative_humidity_2m"),
            "precipitation_mm": weather_data.get("precipitation"),
            "rain_mm": weather_data.get("rain"),
            "weather_code": weather_data.get("weather_code"),
            "wind_speed_kmph": weather_data.get("wind_speed_10m")
        }

        pd.DataFrame([weather_row]).to_csv(
            weather_path,
            mode="a",
            header=not weather_path.exists(),
            index=False
        )

    except Exception as error:
        st.warning(f"Weather update failed: {error}")


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Traffic Intelligence System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #0b1220;
    color: #f1f5f9;
}
[data-testid="stSidebar"] {
    background-color: #111c30;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
h1, h2, h3 {
    color: #f8fafc;
}
p, label, span {
    color: #cbd5e1;
}
div[data-testid="stMetric"] {
    background-color: #17263d;
    border: 1px solid #263b59;
    padding: 18px;
    border-radius: 12px;
}
div[data-testid="stMetricValue"] {
    color: #38bdf8;
}
.stButton button {
    background-color: #0284c7;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: bold;
}
.stButton button:hover {
    background-color: #0369a1;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# --------------------------------------------------
# LOAD HISTORICAL DATA
# --------------------------------------------------

@st.cache_data
def load_data():

    if DATA_PATH.exists():
        return pd.read_csv(
            DATA_PATH,
            keep_default_na=False
        )

    return pd.DataFrame()


data = load_data()


# --------------------------------------------------
# LOAD LIVE DATA
# --------------------------------------------------

def load_live_data():

    traffic = pd.DataFrame()
    weather = pd.DataFrame()
    combined = pd.DataFrame()

    multi_traffic_path = (
        PROJECT_FOLDER / "outputs" /
        "live_traffic_multi_location.csv"
    )

    # Load multi-location traffic data
    if multi_traffic_path.exists():

        traffic = pd.read_csv(multi_traffic_path)

        # Keep only the two configured monitoring locations
        valid_locations = [
            "Existing Nagpur Monitoring Point",
            "Chhatrapati Square - Wardha Road"
        ]

        if "location_name" in traffic.columns:

            traffic = traffic[
                traffic["location_name"].isin(valid_locations)
            ].copy()

    # Fallback to the original traffic file
    elif LIVE_TRAFFIC_PATH.exists():

        traffic = pd.read_csv(LIVE_TRAFFIC_PATH)

    # Load weather data
    if LIVE_WEATHER_PATH.exists():

        weather = pd.read_csv(LIVE_WEATHER_PATH)

    # Load combined data if available
    if COMBINED_LIVE_PATH.exists():

        combined = pd.read_csv(COMBINED_LIVE_PATH)

    return traffic, weather, combined


# --------------------------------------------------
# CONGESTION DETECTION
# --------------------------------------------------

def get_congestion_level(
    current_speed,
    free_flow_speed,
    road_closure
):

    if str(road_closure).strip().lower() in [
        "true", "1", "yes"
    ]:
        return "Road Closed"

    if pd.isna(current_speed) or pd.isna(free_flow_speed):
        return "Insufficient Data"

    if free_flow_speed <= 0:
        return "Insufficient Data"

    reduction = (
        (free_flow_speed - current_speed)
        / free_flow_speed
    ) * 100

    if reduction <= 10:
        return "Low Congestion"

    elif reduction <= 30:
        return "Moderate Congestion"

    elif reduction <= 50:
        return "High Congestion"

    else:
        return "Severe Congestion"


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("🚦 Traffic Intelligence")

st.sidebar.caption("Machine Learning Based Prediction")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Traffic Prediction",
        "Model Performance",
        "Feature Importance",
        "Live Traffic & Weather",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.caption("M.Tech Project | 2026")


# --------------------------------------------------
# DASHBOARD PAGE
# --------------------------------------------------

if page == "Dashboard":

    st.title("🚦 Traffic Intelligence Dashboard")

    st.write(
        "Traffic Volume Analysis and Congestion Prediction"
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Dataset Records",
        f"{len(data):,}" if not data.empty else "N/A"
    )

    col2.metric(
        "Prediction Model",
        "Random Forest"
    )

    col3.metric(
        "Input Features",
        "12"
    )

    col4.metric(
        "Prediction Type",
        "Historical ML"
    )

    st.divider()

    st.subheader("📊 Historical Traffic Overview")

    if not data.empty and "traffic_volume" in data.columns:

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Average Traffic Volume",
                f"{data['traffic_volume'].mean():,.0f}"
            )

        with c2:

            st.metric(
                "Maximum Traffic Volume",
                f"{data['traffic_volume'].max():,.0f}"
            )

        if "date_time" in data.columns:

            data["date_time"] = pd.to_datetime(
                data["date_time"],
                errors="coerce"
            )

            daily_data = (
                data.dropna(subset=["date_time"])
                .set_index("date_time")["traffic_volume"]
                .resample("D")
                .mean()
                .dropna()
            )

            st.markdown("**Daily Average Traffic Volume**")

            daily_chart = (
                daily_data.rename("Average Traffic")
                .rename_axis("Date")
                .reset_index()
            )

            chart = alt.Chart(daily_chart).mark_line(
                color="#38bdf8",
                strokeWidth=2
            ).encode(
                x=alt.X(
                    "Date:T",
                    title="Date",
                    axis=alt.Axis(
                        format="%b %Y",
                        labelAngle=-45
                    )
                ),

                y=alt.Y(
                    "Average Traffic:Q",
                    title="Vehicles",
                    scale=alt.Scale(zero=True)
                ),

                tooltip=[
                    "Date:T",
                    "Average Traffic:Q"
                ]

            ).properties(height=350)

            st.altair_chart(
                chart,
                use_container_width=True
            )

            st.markdown("**Traffic Volume Distribution**")

            if "hour" in data.columns:

                st.bar_chart(
                    data.groupby("hour")["traffic_volume"].mean()
                )

            else:

                st.bar_chart(
                    data.groupby(
                        data["date_time"].dt.hour
                    )["traffic_volume"].mean()
                )

    else:

        st.info("Historical dataset is not available.")

    st.info(
        "Historical traffic analysis uses the Metro Interstate "
        "Traffic Volume dataset. Live Nagpur traffic monitoring "
        "is available on the Live Traffic & Weather page."
    )


# --------------------------------------------------
# TRAFFIC PREDICTION PAGE
# --------------------------------------------------

elif page == "Traffic Prediction":

    st.title("🔮 Traffic Volume Prediction")

    st.write(
        "Enter traffic and weather conditions to estimate traffic volume."
    )

    st.divider()

    st.subheader("Time Information")

    col1, col2 = st.columns(2)

    with col1:

        hour = st.slider(
            "Hour of the Day",
            0, 23, 8
        )

        day = st.number_input(
            "Day of Month",
            1, 31, 23
        )

        month = st.selectbox(
            "Month",
            list(range(1, 13)),
            index=10
        )

    with col2:

        day_of_week = st.selectbox(
            "Day of Week (0 = Monday)",
            list(range(7)),
            index=0
        )

        holiday = st.selectbox(
            "Holiday",
            ["None", "Holiday"]
        )

    st.divider()

    st.subheader("Weather Information")

    col3, col4 = st.columns(2)

    with col3:

        temp = st.number_input(
            "Temperature (Kelvin)",
            200.0, 330.0, 288.5
        )

        rain = st.number_input(
            "Rainfall (mm)",
            0.0, 1000.0, 0.0
        )

        snow = st.number_input(
            "Snowfall (mm)",
            0.0, 100.0, 0.0
        )

    with col4:

        clouds = st.slider(
            "Cloud Coverage (%)",
            0, 100, 40
        )

        weather = st.selectbox(
            "Weather Condition",
            [
                "Clouds", "Clear", "Rain", "Drizzle",
                "Mist", "Snow", "Fog", "Haze",
                "Thunderstorm", "Smoke"
            ]
        )

        description = st.selectbox(
            "Weather Description",
            [
                "scattered clouds", "overcast clouds",
                "broken clouds", "few clouds", "clear sky",
                "light rain", "moderate rain", "mist",
                "haze", "fog", "light snow",
                "heavy intensity rain", "thunderstorm"
            ]
        )

    is_weekend = 1 if day_of_week >= 5 else 0

    st.divider()

    if st.button(
        "🚀 Generate Traffic Prediction",
        use_container_width=True
    ):

        input_data = pd.DataFrame([{
            "holiday": holiday,
            "temp": temp,
            "rain_1h": rain,
            "snow_1h": snow,
            "clouds_all": clouds,
            "weather_main": weather,
            "weather_description": description,
            "hour": hour,
            "day": day,
            "month": month,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend
        }])

        prediction = max(
            0,
            model.predict(input_data)[0]
        )

        st.subheader("Prediction Results")

        r1, r2 = st.columns(2)

        r1.metric(
            "Predicted Traffic Volume",
            f"{prediction:,.0f} vehicles"
        )

        if prediction < 2000:

            status = "Low"

            st.success(
                "🟢 Low Traffic Congestion"
            )

        elif prediction < 4000:

            status = "Medium"

            st.warning(
                "🟡 Medium Traffic Congestion"
            )

        else:

            status = "High"

            st.error(
                "🔴 High Traffic Congestion"
            )

        r2.metric(
            "Congestion Level",
            status
        )

        st.progress(
            min(prediction / 7280, 1.0)
        )

        st.caption(
            "This prediction uses the historical US traffic dataset. "
            "It is not a live Nagpur traffic-volume prediction."
        )


# --------------------------------------------------
# MODEL PERFORMANCE PAGE
# --------------------------------------------------

elif page == "Model Performance":

    st.title("📈 Model Performance Analysis")

    st.write(
        "Evaluation of machine learning algorithms."
    )

    st.divider()

    if COMPARISON_PATH.exists():

        results = pd.read_csv(COMPARISON_PATH)

        st.dataframe(
            results,
            use_container_width=True
        )

        st.subheader("Model Comparison")

        model_column = results.columns[0]

        for metric in ["MAE", "RMSE", "R2"]:

            if metric in results.columns:

                st.markdown(
                    f"### {metric} Comparison"
                )

                chart_data = results.set_index(
                    model_column
                )[[metric]]

                st.bar_chart(chart_data)

                if metric in ["MAE", "RMSE"]:

                    st.caption(
                        "Lower values indicate better performance."
                    )

                else:

                    st.caption(
                        "Higher values indicate better performance."
                    )

    else:

        st.warning(
            "Model comparison results are not available."
        )

    st.markdown("### Best Random Forest Model")

    st.write("MAE: 275.87")
    st.write("RMSE: 489.85")
    st.write("R² Score: 0.9380")

    st.caption(
        "These metrics are from the hourly model comparison experiment. "
        "Other time-series models use different evaluation resolutions."
    )


# --------------------------------------------------
# FEATURE IMPORTANCE PAGE
# --------------------------------------------------

elif page == "Feature Importance":

    st.title("🎯 Feature Importance Analysis")

    st.write(
        "Feature importance indicates how much the trained "
        "Random Forest model relies on each input feature."
    )

    st.divider()

    if IMPORTANCE_PATH.exists():

        st.image(
            str(IMPORTANCE_PATH),
            caption="Random Forest Feature Importance",
            use_container_width=True
        )

    else:

        st.warning(
            "Feature importance chart is not available."
        )


# --------------------------------------------------
# LIVE TRAFFIC AND WEATHER PAGE
# --------------------------------------------------

elif page == "Live Traffic & Weather":

    @st.fragment(run_every="60s")
    def live_traffic_weather_dashboard():

        st.title("🌐 Live Traffic & Weather Monitoring")

        st.write(
            "Automatically refreshed traffic and weather observations "
            "for two selected Nagpur monitoring points."
        )

        st.divider()

        # Collect fresh API data
        collect_live_api_data()

        # Load latest traffic and weather data
        traffic, weather, combined = load_live_data()

        if traffic.empty:

            st.warning(
                "No live traffic records are available."
            )

        else:

            traffic["timestamp_utc"] = pd.to_datetime(
                traffic["timestamp_utc"],
                errors="coerce",
                utc=True
            )

            traffic = traffic.dropna(
                subset=["timestamp_utc"]
            ).sort_values("timestamp_utc")

            # Get latest record for each monitoring location
            latest_by_location = (
                traffic.groupby(
                    "location_name",
                    sort=False
                ).tail(1)
            )

            st.subheader(
                "🚗 Traffic Status by Location"
            )

            for _, location_row in latest_by_location.iterrows():

                current_speed = pd.to_numeric(
                    location_row["current_speed_kmph"],
                    errors="coerce"
                )

                free_flow_speed = pd.to_numeric(
                    location_row["free_flow_speed_kmph"],
                    errors="coerce"
                )

                congestion = get_congestion_level(
                    current_speed,
                    free_flow_speed,
                    location_row["road_closure"]
                )

                st.markdown(
                    f"### 📍 {location_row['location_name']}"
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Current Speed",
                    f"{current_speed:.1f} km/h"
                    if pd.notna(current_speed)
                    else "N/A"
                )

                c2.metric(
                    "Free-flow Speed",
                    f"{free_flow_speed:.1f} km/h"
                    if pd.notna(free_flow_speed)
                    else "N/A"
                )

                c3.metric(
                    "Congestion Level",
                    congestion
                )

                st.divider()

            # Monitoring map
            st.subheader(
                "📍 Nagpur Traffic Monitoring Map"
            )

            map_data = latest_by_location[
                ["latitude", "longitude"]
            ].copy()

            map_data["latitude"] = pd.to_numeric(
                map_data["latitude"],
                errors="coerce"
            )

            map_data["longitude"] = pd.to_numeric(
                map_data["longitude"],
                errors="coerce"
            )

            map_data = map_data.dropna()

            if not map_data.empty:

                st.map(
                    map_data,
                    zoom=12
                )

            st.caption(
                "Monitoring locations: Existing Nagpur point "
                "and Chhatrapati Square, Wardha Road."
            )

            st.divider()

            # Weather information
            st.subheader(
                "☁️ Current Weather Conditions"
            )

            if weather.empty:

                st.warning(
                    "No weather records are available."
                )

            else:

                weather["timestamp"] = pd.to_datetime(
                    weather["timestamp"],
                    errors="coerce",
                    utc=True
                )

                weather = weather.dropna(
                    subset=["timestamp"]
                ).sort_values("timestamp")

                latest_weather = weather.iloc[-1]

                w1, w2, w3, w4 = st.columns(4)

                w1.metric(
                    "Temperature",
                    f"{latest_weather['temperature_c']} °C"
                )

                w2.metric(
                    "Humidity",
                    f"{latest_weather['humidity_percent']}%"
                )

                w3.metric(
                    "Rainfall",
                    f"{latest_weather['rain_mm']} mm"
                )

                w4.metric(
                    "Wind Speed",
                    f"{latest_weather['wind_speed_kmph']} km/h"
                )

                st.caption(
                    "Latest weather observation: "
                    + str(latest_weather["timestamp"])
                )

                st.markdown(
                    "**Weather Reading History**"
                )

                weather_chart = weather.set_index(
                    "timestamp"
                )[
                    [
                        "temperature_c",
                        "humidity_percent"
                    ]
                ]

                st.line_chart(weather_chart)

            st.divider()

            # Traffic speed history
            st.subheader(
                "📊 Live Traffic Speed History"
            )

            traffic_chart = traffic.set_index(
                "timestamp_utc"
            )[
                [
                    "current_speed_kmph",
                    "free_flow_speed_kmph"
                ]
            ]

            st.line_chart(traffic_chart)

            st.divider()

            # Combined records
            st.subheader(
                "📁 Combined Traffic and Weather Records"
            )

            if not combined.empty:

                st.dataframe(
                    combined,
                    use_container_width=True
                )

            else:

                st.info(
                    "Combined data is not available yet."
                )

            st.caption(
                "Traffic speed comes from TomTom. Weather observations "
                "come from Open-Meteo. Traffic monitoring covers two "
                "selected Nagpur points. Weather is collected at the "
                "reference monitoring point."
            )

        st.caption(
            "Automatic refresh interval: 60 seconds"
        )

    live_traffic_weather_dashboard()


# --------------------------------------------------
# ABOUT PROJECT PAGE
# --------------------------------------------------

elif page == "About Project":

    st.title("📚 About the Project")

    st.markdown("""
    ### Real-Time Traffic Congestion Prediction

    **Project Type:** Machine Learning Based Traffic Prediction

    **Academic Level:** M.Tech

    ### Project Objectives

    - Analyze historical traffic volume data.
    - Process traffic and weather information.
    - Develop machine learning and deep learning models.
    - Predict traffic volume.
    - Compare model performance using evaluation metrics.
    - Collect live traffic flow information.
    - Monitor weather conditions.
    - Detect congestion using live traffic speeds.
    - Visualize information through an interactive dashboard.

    ### Algorithms Implemented

    - Random Forest
    - Decision Tree
    - Gradient Boosting
    - Support Vector Regression (SVR)
    - ARIMA
    - Prophet
    - Long Short-Term Memory (LSTM)

    ### Evaluation Metrics

    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - R² Score
    - Time Series Cross-Validation

    ### Data Sources

    - Metro Interstate Traffic Volume historical dataset.
    - TomTom Traffic Flow API.
    - Open-Meteo Weather API.

    ### Current System Limitations

    - Historical traffic-volume training data is from the United States.
    - Live traffic monitoring covers two selected Nagpur points.
    - Live traffic speed is not the same as vehicle-count volume.
    - Current congestion levels are calculated from speed reduction.
    - Live weather is model-based weather data, not a physical sensor reading.
    - Weather observations are collected at one reference coordinate.
    - City-wide traffic monitoring and additional data sources remain to be integrated.
    """)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "M.Tech Project | Traffic Congestion Prediction | "
    "Machine Learning and Deep Learning"
)

