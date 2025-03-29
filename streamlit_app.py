%%writefile app.py

import streamlit as st
import numpy as np
import tensorflow as tf
import joblib  # Use joblib instead of pickle
import time

# Load the CNN feature extractor
@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model("cnn_feature_extractor.h5")

cnn_model = load_cnn_model()

# Load the Random Forest classifier
@st.cache_resource
def load_rf_model():
    return joblib.load("rf_model.pkl")  # Fixed: Use joblib

rf_model = load_rf_model()

# Load the saved scaler
@st.cache_resource 
def load_scaler():
    return joblib.load("scaler.pkl")  # Fixed: Use joblib

scaler = load_scaler()

# Streamlit UI
st.title("DDoS Attack Prediction System (CNN + RF)")

# Generate synthetic network traffic sample
def generate_traffic():
    return np.array([[np.random.randint(0, 65535),    # Destination Port
                      np.random.randint(100, 10000),  # Flow Duration (ms)
                      np.random.randint(50, 1500),    # Fwd Packet Length Mean
                      np.random.randint(50, 1500),    # Bwd Packet Length Mean
                      np.random.uniform(1000, 100000), # Flow Bytes/s
                      np.random.uniform(10, 1000),    # Flow Packets/s
                      np.random.uniform(1, 500)]],    # Flow IAT Mean
                    dtype=np.float32)

# Button to trigger detection
if st.button("Start Detection"):
    st.write("### Real-Time Traffic Detection:")
    
    input_data = generate_traffic()
    scaled_input = scaler.transform(input_data)
    cnn_input = scaled_input.reshape(1, 7, 1)
    cnn_features = cnn_model.predict(cnn_input)
    prediction = rf_model.predict(cnn_features)
    is_ddos = prediction[0] == 1

    # Display traffic data
    st.write(
        f"""
        **Traffic Data:**
        - **Destination Port:** {int(input_data[0][0])}
        - **Flow Duration (ms):** {int(input_data[0][1])}
        - **Fwd Packet Length Mean:** {int(input_data[0][2])}
        - **Bwd Packet Length Mean:** {int(input_data[0][3])}
        - **Flow Bytes/s:** {input_data[0][4]:,.2f}
        - **Flow Packets/s:** {input_data[0][5]:,.2f}
        - **Flow IAT Mean:** {input_data[0][6]:,.2f}
        """
    )

    result = "**🚨 DDoS Attack Detected!**" if is_ddos else "✅ Normal Traffic"
    color = "red" if is_ddos else "green"

    st.markdown(
        f"<div style='background-color:{color}; padding:10px; border-radius:10px;'>"
        f"<h2 style='color:white; text-align:center;'>{result}</h2>"
        f"</div>", unsafe_allow_html=True
    )
