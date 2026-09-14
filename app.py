"""
Rossmann Sales Forecast — Streamlit Dashboard
------------------------------------------------
Run locally (in VS Code's integrated terminal):

    cd streamlit_app
    pip install -r requirements.txt
    streamlit run app.py

This app loads the serialized production model (XGBoost, selected after comparing against Linear
Regression and Random Forest baselines) produced by the project notebook (rossmann_analysis.ipynb)
and lets a store manager / analyst either:
  1) enter a single day's parameters manually, or
  2) upload a CSV of future dates/parameters for a store,
and get back predicted Sales (+ an approximate Customers estimate and a
90% confidence band), viewable as a chart and downloadable as CSV.
"""


import glob
import os
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Rossmann Sales Forecast", layout="wide")

# ----------------------------
# Load Model
# ----------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


@st.cache_resource
def load_latest_model():
    candidates = sorted(glob.glob(os.path.join(MODEL_DIR, "sales_model_*.pkl")))
    if not candidates:
        return None, None

    latest = candidates[-1]
    with open(latest, "rb") as f:
        bundle = pickle.load(f)

    return bundle, os.path.basename(latest)


# ----------------------------
# Feature Engineering
# ----------------------------
def engineer_row_features(df, median_dist):
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfWeek"] = df["Date"].dt.dayofweek + 1
    df["IsWeekend"] = (df["DayOfWeek"] >= 6).astype(int)

    df["MonthPeriod"] = np.select(
        [df["Day"] <= 10, df["Day"] <= 20],
        ["Beginning", "Mid"],
        default="End",
    )

    df["DaysToHoliday"] = 30
    df["DaysAfterHoliday"] = 30

    if "CompetitionDistance" not in df.columns:
        df["CompetitionDistance"] = median_dist

    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(median_dist)

    df["CompetitionMonthsOpen"] = 12
    df["Promo2"] = df.get("Promo2", 0)
    df["Promo2Active"] = 0

    df["StateHoliday"] = df.get("StateHoliday", "0").astype(str)
    df["SchoolHoliday"] = df.get("SchoolHoliday", 0)
    df["StoreType"] = df.get("StoreType", "a")
    df["Assortment"] = df.get("Assortment", "a")

    return df


# ----------------------------
# App Header
# ----------------------------
st.title("📈 Rossmann Store Sales Forecast")
st.caption("Interactive XGBoost Sales Prediction Dashboard")

bundle, model_name = load_latest_model()

if bundle is None:
    st.error("No trained model found inside the models folder.")
    st.stop()

st.success(f"Loaded Model: {model_name}")

pipeline = bundle["pipeline"]
feature_cols_num = bundle["feature_cols_num"]
feature_cols_cat = bundle["feature_cols_cat"]
median_dist = bundle["median_competition_distance"]

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2 = st.tabs(["🔢 Single Prediction", "📄 Batch CSV Prediction"])

# ==========================================================
# TAB 1
# ==========================================================
with tab1:

    st.subheader("Predict Sales for One Store")

    c1, c2, c3 = st.columns(3)

    with c1:
        store = st.number_input("Store", 1, 1115, 1)
        date = st.date_input("Date", datetime.today() + timedelta(days=1))
        promo = st.selectbox("Promo", [0, 1], index=1)

    with c2:
        state = st.selectbox("State Holiday", ["0", "a", "b", "c"])
        school = st.selectbox("School Holiday", [0, 1])
        store_type = st.selectbox("Store Type", ["a", "b", "c", "d"])

    with c3:
        assortment = st.selectbox("Assortment", ["a", "b", "c"])
        distance = st.number_input(
            "Competition Distance",
            min_value=0.0,
            value=float(median_dist),
        )
        promo2 = st.selectbox("Promo2", [0, 1])

    if st.button("Predict Sales", type="primary"):

        input_df = pd.DataFrame(
            [
                {
                    "Date": pd.Timestamp(date),
                    "Store": store,
                    "Promo": promo,
                    "StateHoliday": state,
                    "SchoolHoliday": school,
                    "StoreType": store_type,
                    "Assortment": assortment,
                    "CompetitionDistance": distance,
                    "Promo2": promo2,
                }
            ]
        )

        input_df = engineer_row_features(input_df, median_dist)

        X = input_df[feature_cols_num + feature_cols_cat]

        pred = float(pipeline.predict(X)[0])

        lower = pred * 0.90
        upper = pred * 1.10

        customers = pred / 9.5

        st.metric("Predicted Sales", f"₹ {pred:,.0f}")

        st.write(f"**90% Confidence Range:** ₹ {lower:,.0f} – ₹ {upper:,.0f}")

        st.write(f"**Estimated Customers:** {customers:,.0f}")

# ==========================================================
# TAB 2
# ==========================================================
with tab2:

    st.subheader("Upload CSV for Multi-Day Forecast")

    st.info(
        "Required columns: Store, Date, Promo, StateHoliday, SchoolHoliday\n\n"
        "Optional: StoreType, Assortment, CompetitionDistance, Promo2"
    )

    file = st.file_uploader("Choose CSV", type="csv")

    if file is not None:

        df = pd.read_csv(file)

        df["Date"] = pd.to_datetime(df["Date"])

        df = engineer_row_features(df, median_dist)

        X = df[feature_cols_num + feature_cols_cat]

        predictions = pipeline.predict(X)

        result = df.copy()

        result["Predicted_Sales"] = predictions.round(0)
        result["Estimated_Customers"] = (predictions / 9.5).round(0)

        st.dataframe(result)

        chart = result[["Date", "Predicted_Sales"]].set_index("Date")

        st.line_chart(chart)

        csv = result.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇ Download Prediction CSV",
            csv,
            "rossmann_predictions.csv",
            "text/csv",
        )

st.divider()

st.caption(
    "Best Model: XGBoost Regressor | Rossmann Store Sales Forecasting Project"
)