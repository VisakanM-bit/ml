 
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="🌡️ Temperature Predictor", layout="centered")

st.title("🌡️ Temperature Prediction App")
st.markdown("Enter the weather parameters below to predict **Next_Tmax** and **Next_Tmin**.")

# Input form
input_dict = {}
features = [
    "Present_Tmax", "Present_Tmin", "LDAPS_RHmin", "LDAPS_RHmax", "LDAPS_Tmax_lapse",
    "LDAPS_Tmin_lapse", "LDAPS_WS", "LDAPS_LH", "LDAPS_CC1", "LDAPS_CC2",
    "LDAPS_CC3", "LDAPS_CC4", "LDAPS_PPT1", "LDAPS_PPT2", "LDAPS_PPT3", "LDAPS_PPT4",
    "lat", "lon", "DEM", "Slope", "Solar radiation"
]

for feature in features:
    input_dict[feature] = st.number_input(feature, value=0.00, format="%.2f")

# Prediction
if st.button("Predict"):
    input_df = pd.DataFrame([input_dict])

    # Load saved model and scalers
    model = joblib.load("temperature_model.pkl")
    imputer = joblib.load("imputer.pkl")
    scaler = joblib.load("scaler.pkl")

    # Apply transformations
    input_df = imputer.transform(input_df)
    input_df = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_df)
    st.success(f"🌞 Predicted Next_Tmax: {prediction[0][0]:.2f} °C")
    st.success(f"🌙 Predicted Next_Tmin: {prediction[0][1]:.2f} °C")
