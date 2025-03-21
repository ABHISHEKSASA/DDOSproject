import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import os

# ==============================
# ✅ Model and Scaler Loading
# ==============================

# Cache model and scaler loading
@st.cache_resource
def load_model():
    model_path = "ddos_cnn_model.h5"  # Ensure correct path
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        st.success("✅ Model Loaded Successfully!")
        return model
    else:
        st.error("❌ Model file not found!")
        return None

@st.cache_resource
def load_scaler():
    scaler_path = "scaler.pkl"  # Ensure correct path
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        st.success("✅ Scaler Loaded Successfully!")
        return scaler
    else:
        st.error("❌ Scaler file not found!")
        return None

cnn_model = load_model()
scaler = load_scaler()

# ==============================
# ✅ Streamlit UI
# ==============================

st.title("🔄 DDoS Attack Prediction System")
st.write("### **Enter Network Traffic Features:**")

# User Input Section
destination_port = st.number_input("Destination Port (e.g., 80, 443, 22, 53)", min_value=0, max_value=65535, value=80)
flow_duration = st.number_input("Flow Duration (in ms)", min_value=1, value=1000)
fwd_packet_length_mean = st.number_input("Fwd Packet Length Mean", min_value=1, value=500)
bwd_packet_length_mean = st.number_input("Bwd Packet Length Mean", min_value=1, value=500)
flow_bytes_per_s = st.number_input("Flow Bytes/s", min_value=0.1, value=50000.0)
flow_packets_per_s = st.number_input("Flow Packets/s", min_value=0.1, value=50.0)
flow_iat_mean = st.number_input("Flow IAT Mean", min_value=0.1, value=100.0)

# Collect user input into a NumPy array
input_data = np.array([[destination_port, flow_duration, fwd_packet_length_mean, 
                        bwd_packet_length_mean, flow_bytes_per_s, flow_packets_per_s, flow_iat_mean]],
                      dtype=np.float32)

# ==============================
# ✅ Prediction Logic
# ==============================

if st.button("🚀 Predict DDoS Attack"):
    if cnn_model is None or scaler is None:
        st.error("❌ Model or scaler not loaded properly!")
    else:
        # Scale input data
        try:
            scaled_input_data = scaler.transform(input_data)
        except Exception as e:
            st.error(f"Scaling Error: {e}")
            scaled_input_data = input_data  # Fallback in case of scaling issue

        # Log for debugging
        st.write("### 📊 Raw Input Data:", input_data)
        st.write("### 🔥 Scaled Input Data:", scaled_input_data)

        # Adjust shape based on model input
        model_input_shape = cnn_model.input_shape
        st.write("### 🔧 Model Input Shape:", model_input_shape)

        if len(model_input_shape) == 3:
            scaled_input_data = scaled_input_data.reshape(1, 7, 1)  # Reshape for CNN
        elif len(model_input_shape) == 2:
            scaled_input_data = scaled_input_data.reshape(1, 7)

        # Make prediction
        prediction = cnn_model.predict(scaled_input_data)
        st.write("### ⚙️ Raw Prediction Probability:", prediction[0][0])

        # Interpret prediction
        threshold = 0.3  # Adjust as needed
        if prediction[0][0] > threshold:
            result = "🚀 **DDoS Attack Detected!**"
            st.error(result)
        else:
            result = "✅ **Normal Traffic**"
            st.success(result)

