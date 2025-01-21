import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# Streamlit App
st.title("Vorhersage von Versicherungsansprüchen")
st.markdown("Willkommen! Nutzen Sie dieses Tool, um vorherzusagen, ob ein Versicherungsanspruch geltend gemacht wird.")

# Load the pretrained model
@st.cache_resource
def load_model():
    try:
        return joblib.load("xgb_model.joblib")  # Replace with the correct model file path
    except FileNotFoundError:
        st.error("Das Modell 'xgb_model.joblib' konnte nicht gefunden werden.")
        return None

model = load_model()

# Check if the model is loaded
if model:
    # Top features in correlation to the target variable
    feature_ranges = {
        'policy_tenure': (0, 50),  # Example: 0 to 50 years
        'age_of_policyholder': (18, 100),  # Example: 18 to 100 years
        'is_adjustable_steering': (0, 1),  # Binary input: 0 or 1
        'cylinder': (1, 12)  # Example: 1 to 12 cylinders
    }

    # Input Form for Inference
    st.subheader("Eingabefunktionen für die Vorhersage")
    user_input = {}
    for feature, (min_val, max_val) in feature_ranges.items():
        user_input[feature] = st.number_input(
            f"Wert für {feature} eingeben ({min_val} bis {max_val}):",
            min_value=min_val,
            max_value=max_val,
            value=min_val
        )

    # Predict Button
    if st.button("Vorhersagen"):
        try:
            # Convert user input to a DataFrame
            input_data = pd.DataFrame([user_input])

            # Scale the input data
            scaler = MinMaxScaler()
            input_data_scaled = scaler.fit_transform(input_data)

            # Make prediction
            prediction = model.predict(input_data_scaled)

            # Display result
            result = (
                "Anspruch"
                if prediction[0] == 1
                else "Keinen Anspruch"
            )
            st.subheader(f"Vorhersage: {result}")
        except Exception as e:
            st.error(f"Fehler bei der Vorhersage: {e}")
else:
    st.warning("Bitte laden Sie ein gültiges Modell hoch, um Vorhersagen durchzuführen.")

# Footer with Website and GitHub links
st.markdown(
    """
    <style>
    footer {
        visibility: hidden;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #222121;
        text-align: left;
        padding: 10px;
        font-size: 14px;
    }
    </style>
    <div class="footer">
        <p>Besuchen Sie meine <a href="https://example.com" target="_blank">Website</a> | 
         <a href="https://github.com/your-github" target="_blank">Project Code</a></p>
    </div>
    """,
    unsafe_allow_html=True
)
