import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# Load the CNN model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("ddos_cnn_model (2).h5")  # Ensure correct path
    return model

cnn_model = load_model()

# Load the saved scaler
@st.cache_resource
def load_scaler():
    with open("scaler (2).pkl", "rb") as f:
        scaler = pickle.load(f)
    return scaler

scaler = load_scaler()

st.title("🔄 DDoS Attack Prediction System")

# User Input Section
st.write("### **Enter Network Traffic Features:**")
destination_port = st.number_input(min_value=0, max_value=65535, value=80)
flow_duration = st.number_input("Flow Duration (in ms)", min_value=1, value=1000)
fwd_packet_length_mean = st.number_input("Fwd Packet Length Mean", min_value=1, value=500)
bwd_packet_length_mean = st.number_input("Bwd Packet Length Mean", min_value=1, value=500)
flow_bytes_per_s = st.number_input("Flow Bytes/s", min_value=0.1, value=50000.0)
flow_packets_per_s = st.number_input("Flow Packets/s", min_value=0.1, value=50.0)
flow_iat_mean = st.number_input("Flow IAT Mean", min_value=0.1, value=100.0)

# Collect user input into a NumPy array
input_data = np.array([[destination_port, flow_duration, fwd_packet_length_mean, 
                        bwd_packet_length_mean, flow_bytes_per_s, flow_packets_per_s, flow_iat_mean]], dtype=np.float32)

# Predict Button
if st.button("Predict DDoS Attack"):
    # Scale input data
    scaled_input_data = scaler.transform(input_data)

    # Adjust shape based on model input
    model_input_shape = cnn_model.input_shape
    if len(model_input_shape) == 3:
        scaled_input_data = scaled_input_data.reshape(1, 7, 1)  # Reshape for CNN
    elif len(model_input_shape) == 2:
        scaled_input_data = scaled_input_data.reshape(1, 7)

    # Predict using the CNN model
    prediction = cnn_model.predict(scaled_input_data)

    # Print raw probability for debugging

    # Interpret prediction (Adjust threshold if needed)
    threshold = 0.3  # Adjust if necessary (e.g., 0.3 for more sensitivity)
    result = "🚀 **DDoS Attack Detected!**" if prediction[0][0] > threshold else "✅ **Normal Traffic**"

    # Display prediction result
    st.write("### **Prediction:**", result)
