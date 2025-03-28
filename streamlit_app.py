import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import time

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

st.title("🔄 DDoS Attack Prediction System (Automatic Detection)")

# Function to simulate random network traffic data
def generate_traffic_data():
    destination_port = np.random.randint(0, 65535)
    flow_duration = np.random.randint(100, 10000)  # Flow duration in ms
    fwd_packet_length_mean = np.random.randint(50, 1500)
    bwd_packet_length_mean = np.random.randint(50, 1500)
    flow_bytes_per_s = np.random.uniform(1000, 1_000_000)  # Bytes per second
    flow_packets_per_s = np.random.uniform(10, 1000)  # Packets per second
    flow_iat_mean = np.random.uniform(1, 1000)  # Inter-Arrival Time mean

    return np.array([[destination_port, flow_duration, fwd_packet_length_mean, 
                      bwd_packet_length_mean, flow_bytes_per_s, 
                      flow_packets_per_s, flow_iat_mean]], dtype=np.float32)

# Automatic prediction loop
placeholder = st.empty()

# Threshold for DDoS detection
threshold = 0.3  # Adjust sensitivity if needed

# Continuous monitoring
st.write("### **Real-Time Network Traffic Detection:**")

# Start detection loop
if st.button("Start Detection"):
    with st.spinner("Monitoring traffic... Press STOP to end."):
        try:
            while True:
                # Simulate traffic data
                input_data = generate_traffic_data()

                # Scale input data
                scaled_input_data = scaler.transform(input_data)

                # Reshape for CNN input
                model_input_shape = cnn_model.input_shape
                if len(model_input_shape) == 3:
                    scaled_input_data = scaled_input_data.reshape(1, 7, 1)
                elif len(model_input_shape) == 2:
                    scaled_input_data = scaled_input_data.reshape(1, 7)

                # Make prediction
                prediction = cnn_model.predict(scaled_input_data)
                result = "🚀 **DDoS Attack Detected!**" if prediction[0][0] > threshold else "✅ **Normal Traffic**"

                # Display real-time result
                placeholder.write(f"### **Prediction:** {result}")
                time.sleep(2)  # Update every 2 seconds

        except KeyboardInterrupt:
            st.write("🛑 Monitoring stopped.")
