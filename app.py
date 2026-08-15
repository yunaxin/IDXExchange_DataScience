import streamlit as st
import joblib
import numpy as np
import pandas as pd

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("Home Price Predictor")
st.write("Enter the property details below to get an estimated closing price.")

living_area = st.number_input("Living Area (sq ft)", min_value=200, max_value=15000, value=1800, step=50)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
lot_size = st.number_input("Lot Size (sq ft)", min_value=500, max_value=100000, value=6000, step=100)

if st.button("Predict Price"):
    input_df = pd.DataFrame([{
        "LivingArea": living_area,
        "BedroomsTotal": bedrooms,
        "BathroomsTotalInteger": bathrooms,
        "LotSizeSquareFeet": lot_size,
    }])
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Closing Price: ${prediction:,.0f}")