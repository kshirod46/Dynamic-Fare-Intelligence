from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
import zlib
from urllib.request import urlopen

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from joblib import load

st.set_page_config(
    page_title="Dynamic Fare Intelligence | Indian Ride-Hailing",
    page_icon="🚕",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "fare_model.joblib"
MODEL_URL = (
    "https://github.com/kshirod46/Dynamic-Fare-Intelligence/"
    "releases/download/v1.0.0/fare_model.joblib"
)

FEATURES = [
    "Pickup_City",
    "Vehicle_Type",
    "Distance_KM",
    "Duration_Min",
    "Time_of_Day",
    "Weather_Condition",
    "Traffic_Level",
    "Day_of_Week",
    "Is_Prime_Member",
    "Driver_Rating",
]


class ModelArtifactError(RuntimeError):
    """Raised when the downloaded model cannot be loaded."""


@st.cache_resource
def load_artifacts():
    model = None
    if MODEL_PATH.exists():
        try:
            model = load(MODEL_PATH)
        except (EOFError, OSError, ValueError, zlib.error):
            MODEL_PATH.unlink()

    if model is None:
        if not MODEL_URL:
            raise FileNotFoundError(
                "fare_model.joblib is missing. Set MODEL_URL to the public "
                "GitHub Release asset URL."
            )
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            with st.spinner("Downloading the ML model..."):
                with tempfile.NamedTemporaryFile(
                    prefix="fare_model_",
                    suffix=".part",
                    dir=MODEL_DIR,
                    delete=False,
                ) as temp_file:
                    temp_path = Path(temp_file.name)
                    with urlopen(MODEL_URL, timeout=120) as response:
                        shutil.copyfileobj(response, temp_file)

                try:
                    model = load(temp_path)
                except (EOFError, OSError, ValueError, zlib.error) as exc:
                    raise ModelArtifactError(
                        "The GitHub Release model is corrupted. Replace the "
                        "fare_model.joblib release asset with a fresh upload."
                    ) from exc
                os.replace(temp_path, MODEL_PATH)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    feature_meta = json.loads((MODEL_DIR / "feature_metadata.json").read_text())
    results = pd.read_csv(MODEL_DIR / "model_comparison.csv")
    return model, feature_meta, results


def feature_importance_frame(model) -> pd.DataFrame:
    preprocess = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]
    try:
        names = preprocess.get_feature_names_out()
    except Exception:
        return pd.DataFrame(columns=["Feature", "Importance"])

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_)
    else:
        return pd.DataFrame(columns=["Feature", "Importance"])

    values = np.asarray(values).ravel()
    df = pd.DataFrame({"Feature": names, "Importance": values})
    return df.sort_values("Importance", ascending=False).head(12)


try:
    model, feature_meta, results_df = load_artifacts()
except (FileNotFoundError, ModelArtifactError) as exc:
    st.error(str(exc))
    st.stop()

st.title("🚕 Dynamic Fare Intelligence")
st.caption("Indian ride-hailing | ML-powered fare estimation")

best_model = feature_meta["best_model"]
categories = feature_meta.get("categories", {})

tab1, tab2, tab3 = st.tabs(
    ["🔮 Fare Prediction", "📊 Model Evaluation", "ℹ️ Methodology"]
)

with tab1:
    st.subheader("Create a ride quote")

    c1, c2, c3 = st.columns(3)
    with c1:
        city = st.selectbox(
            "Pickup City",
            categories.get("Pickup_City", ["Bangalore", "Mumbai", "Delhi", "Hyderabad", "Chennai", "Kolkata", "Pune", "Ahmedabad"]),
        )
        vehicle = st.selectbox("Vehicle Type", categories.get("Vehicle_Type", ["Bike", "Auto", "Mini", "Sedan", "SUV", "Prime"]))
        distance = st.number_input("Distance (km)", min_value=1.0, max_value=50.0, value=10.0, step=0.5)

    with c2:
        time_of_day = st.selectbox("Time of Day", categories.get("Time_of_Day", ["Early Morning", "Morning", "Afternoon", "Evening", "Night"]))
        weather = st.selectbox("Weather", categories.get("Weather_Condition", ["Clear", "Cloudy", "Rainy", "Foggy", "Stormy"]))
        traffic = st.selectbox("Traffic", categories.get("Traffic_Level", ["Low", "Medium", "High", "Severe"]))

    with c3:
        day = st.selectbox(
            "Day of Week",
            categories.get("Day_of_Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]),
        )
        prime_member = st.selectbox("Prime Member", [0, 1], format_func=lambda x: "Yes" if x else "No")
        rating = st.slider("Driver Rating", 3.0, 5.0, 4.5, 0.1)

    # A transparent, editable proxy for trip duration when the user does not
    # provide route-specific ETA. Production deployment should use routing ETA.
    duration = distance * 3.0

    row = pd.DataFrame(
        {
            "Pickup_City": [city],
            "Vehicle_Type": [vehicle],
            "Distance_KM": [distance],
            "Duration_Min": [duration],
            "Time_of_Day": [time_of_day],
            "Weather_Condition": [weather],
            "Traffic_Level": [traffic],
            "Day_of_Week": [day],
            "Is_Prime_Member": [prime_member],
            "Driver_Rating": [rating],
        }
    )[FEATURES]

    if st.button("Predict Fare", type="primary", use_container_width=True):
        predicted_fare = float(model.predict(row)[0])
        predicted_fare = max(predicted_fare, 0.0)

        st.session_state["predicted_fare"] = predicted_fare

    if "predicted_fare" in st.session_state:
        predicted_fare = st.session_state["predicted_fare"]

        m1, m2, m3 = st.columns(3)
        m1.metric("ML Predicted Fare", f"₹{predicted_fare:,.0f}")
        m2.metric("Selected Model", best_model)
        m3.metric("Distance", f"{distance:.1f} km")

        imp = feature_importance_frame(model)
        if not imp.empty:
            st.subheader("What drives the fare model?")
            fig_imp = px.bar(
                imp.sort_values("Importance"),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top model features",
            )
            fig_imp.update_layout(height=420)
            st.plotly_chart(fig_imp, use_container_width=True)

with tab2:
    st.subheader("Model comparison")
    st.dataframe(
        results_df.style.format({"RMSE": "₹{:.2f}", "MAE": "₹{:.2f}", "R2": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

    fig = px.bar(
        results_df.sort_values("RMSE"),
        x="Model",
        y="RMSE",
        title="Test RMSE — lower is better",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Metrics come from a fixed 80/20 holdout split (random_state=42). "
        "For production, add time-based validation and drift monitoring."
    )

with tab3:
    st.subheader("Methodology")
    st.markdown(
        """
        **Fare model:** the target is `Total_Fare` for completed trips. Categorical
        features are one-hot encoded inside the serialized sklearn pipeline.

        **Model selection:** Ridge Regression, Random Forest, and Gradient Boosting
        are compared using RMSE, MAE, and R². The lowest-RMSE model is saved.

        **Prediction only:** the dashboard estimates the fare for a completed-trip
        style input using the saved machine-learning pipeline.
        """
    )
    st.subheader("Training features")
    st.write(", ".join(feature_meta["features"]))
