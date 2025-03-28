import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import time

# 📥 Load the CNN model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("ddos_cnn_model (2).h5")  # Ensure correct path
    return model

cnn_model = load_model()

# 📥 Load the saved scaler
@st.cache_resource
def load_scaler():
    with open("scaler (2).pkl", "rb") as f:
        scaler = pickle.load(f)
    return scaler

scaler = load_scaler()

# 📊 Load real-world traffic data (CSV format)
@st.cache_resource
def load_traffic_data():
    # Replace with the path to your dataset
    file_path = "ddos_traffic.csv"  # Ensure the CSV contains the required features
    df = pd.read_csv(file_path)
    
    # Select the 7 required features for prediction
    feature_cols = ["Dst Port", "Flow Duration", "Fwd Packet Length Mean", 
                    "Bwd Packet Length Mean", "Flow Bytes/s", "Flow Packets/s", 
                    "Flow IAT Mean"]
    
    # Extract features
    traffic_data = df[feature_cols].values
    return traffic_data

# Load traffic data
traffic_data = load_traffic_data()

# 🎯 Streamlit UI
st.title("🔄 DDoS Attack Prediction System (Real-World Data)")

# UI layout
col1, col2 = st.columns(2)
start_btn = col1.button("▶️ Start Detection")
stop_btn = col2.button("🛑 Stop Detection")

# 💡 Initialize state variables
if 'running' not in st.session_state:
    st.session_state['running'] = False

if start_btn:
    st.session_state['running'] = True

if stop_btn:
    st.session_state['running'] = False

# 🟢 Detection Loop
status_placeholder = st.empty()
traffic_placeholder = st.empty()

# DDoS detection threshold
threshold = 0.3  

if st.session_state['running']:
    st.write("### 🚀 **Real-Time Network Traffic Detection:**")

    normal_count = 0
    ddos_count = 0

    try:
        for i in range(len(traffic_data)):
            if not st.session_state['running']:
                break

            # Get the current sample
            input_data = traffic_data[i].reshape(1, -1)

            # Scale the input data
            scaled_input = scaler.transform(input_data)

            # Reshape for CNN model compatibility
            model_input_shape = cnn_model.input_shape

            if len(model_input_shape) == 3:
                scaled_input = scaled_input.reshape(1, 7, 1)
            elif len(model_input_shape) == 2:
                scaled_input = scaled_input.reshape(1, 7)

            # Make prediction
            prediction = cnn_model.predict(scaled_input)
            
            # Determine if DDoS or normal
            is_ddos = prediction[0][0] > threshold

            # Count occurrences
            if is_ddos:
                ddos_count += 1
            else:
                normal_count += 1

            # Display traffic data
            traffic_placeholder.write(
                f"""
                **Traffic Data:**  
                - 🌐 **Destination Port:** {int(input_data[0][0])}  
                - ⏱️ **Flow Duration (ms):** {int(input_data[0][1])}  
                - 📦 **Fwd Packet Length Mean:** {int(input_data[0][2])}  
                - 📦 **Bwd Packet Length Mean:** {int(input_data[0][3])}  
                - 🔥 **Flow Bytes/s:** {input_data[0][4]:,.2f}  
                - 🚀 **Flow Packets/s:** {input_data[0][5]:,.2f}  
                - ⏲️ **Flow IAT Mean:** {input_data[0][6]:,.2f}
                """
            )

            # Display detection result
            result = "🚀 **DDoS Attack Detected!**" if is_ddos else "✅ **Normal Traffic**"
            color = "red" if is_ddos else "green"

            status_placeholder.markdown(
                f"<div style='background-color:{color}; padding:10px; border-radius:10px;'>"
                f"<h2 style='color:white; text-align:center;'>{result}</h2>"
                f"</div>", unsafe_allow_html=True
            )

            # Display DDoS vs Normal traffic count
            st.write("### 📊 **Traffic Statistics:**")
            st.write(f"- **DDoS Attacks:** {ddos_count}")
            st.write(f"- **Normal Traffic:** {normal_count}")

            # Refresh every second
            time.sleep(1)

    except Exception as e:
        st.error(f"❌ Error: {e}")

