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

# Streamlit UI
st.title("🔄 DDoS Attack Prediction System (Automatic Detection)")

# UI layout
col1, col2 = st.columns(2)
start_btn = col1.button("▶️ Start Detection")
stop_btn = col2.button("🛑 Stop Detection")

# ⚙️ Simulate 50% Normal and 50% DDoS Traffic
def generate_traffic():
    """Simulate equal distribution of normal and DDoS-like network traffic."""
    if np.random.random() < 0.5:  # 50% normal, 50% DDoS
        # Normal traffic
        return np.array([[np.random.randint(0, 65535),     # Destination Port
                          np.random.randint(100, 10000),   # Flow Duration (ms)
                          np.random.randint(50, 1500),     # Fwd Packet Length Mean
                          np.random.randint(50, 1500),     # Bwd Packet Length Mean
                          np.random.uniform(1000, 100000), # Flow Bytes/s (normal)
                          np.random.uniform(10, 100),      # Flow Packets/s
                          np.random.uniform(100, 1000)]],  # Flow IAT Mean
                        dtype=np.float32)
    else:
        # DDoS traffic
        return np.array([[80,               # Destination Port (fixed)
                          10000,            # Long Flow Duration (ms)
                          1400,             # Large Fwd Packet Length Mean
                          1400,             # Large Bwd Packet Length Mean
                          950000.0,         # High Flow Bytes/s
                          980.0,            # High Flow Packets/s
                          5.0]], dtype=np.float32)  # Low IAT (frequent packets)

# Continuous monitoring logic
status_placeholder = st.empty()
traffic_placeholder = st.empty()

# DDoS detection threshold
threshold = 0.3  

# Start/Stop button handling
if start_btn:
    st.session_state['running'] = True

if stop_btn:
    st.session_state['running'] = False

if 'running' not in st.session_state:
    st.session_state['running'] = False

# Detection loop
if st.session_state['running']:
    st.write("### 🚀 **Real-Time Network Traffic Detection:**")

    normal_count = 0
    ddos_count = 0

    try:
        while st.session_state['running']:
            # Generate traffic data
            input_data = generate_traffic()

            # Scale and reshape input for the CNN model
            scaled_input = scaler.transform(input_data)
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

            # Refresh every 2 seconds
            time.sleep(2)

    except Exception as e:
        st.error(f"❌ Error: {e}")

