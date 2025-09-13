# streamlit_app.py
import streamlit as st
import requests

# FastAPI endpoint
FASTAPI_URL = "http://fastapi:8000/predict"

# Streamlit app UI
st.title("Wine Classifier")

# Input fields for the Wine features
alcohol = st.number_input("Alcohol", min_value=0.0, value=13.0)
malic_acid = st.number_input("Malic Acid", min_value=0.0, value=2.0)
color_intensity = st.number_input("Color Intensity", min_value=0.0, value=5.0)
hue = st.number_input("Hue", min_value=0.0, value=1.0)
proline = st.number_input("Proline", min_value=0.0, value=750.0)

# Make prediction when the button is clicked
if st.button("Predict"):
    # Prepare the data for the API request
    input_data = {
        "alcohol": alcohol,
        "malic_acid": malic_acid,
        "color_intensity": color_intensity,
        "hue": hue,
        "proline": proline
    }
    # Send a request to the FastAPI prediction endpoint
    response = requests.post(FASTAPI_URL, json=input_data)
    prediction = response.json()["prediction"]
    # Display the result
    st.success(f"The model predicts wine class: {prediction}")
