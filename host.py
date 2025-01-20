import streamlit as st
import pandas as pd
import numpy as np
import joblib  # For loading/saving models
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Streamlit App
st.title("Insurance Claims Prediction")

# Load the pretrained model
@st.cache_resource
def load_model():
    return joblib.load("xgb_model.joblib")  # Replace with the correct model file path

model = load_model()

# Define feature names (should match the model's training features)
feature_names = ['policy_tenure', 'age_of_policyholder', 'is_adjustable_steering', 'cylinder']

# Input Form for Inference
st.subheader("Input Features for Prediction")
user_input = {}
for feature in feature_names:
    user_input[feature] = st.number_input(f"Enter value for {feature}:", value=0.0)

# Predict Button
if st.button("Predict"):
    try:
        # Convert user input to a DataFrame
        input_data = pd.DataFrame([user_input])

        # Scale the input data (assumes MinMaxScaler was used during training)
        scaler = MinMaxScaler()
        input_data_scaled = scaler.fit_transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)

        # Display result
        result = "Claim" if prediction[0] == 1 else "No Claim"
        st.subheader(f"Prediction: {result}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
