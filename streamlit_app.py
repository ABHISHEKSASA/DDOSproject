import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go
import tensorflow as tf
import pickle

# Load the CNN model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("ddos_cnn_model (2).h5")  # Ensure the correct path
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

cnn_model = load_model()

# Load the scaler
@st.cache_resource
def load_scaler():
    try:
        with open("scaler (2).pkl", "rb") as f:
            scaler = pickle.load(f)
        return scaler
    except Exception as e:
        st.error(f"Failed to load scaler: {e}")
        return None

scaler = load_scaler()

# App layout and styling
st.set_page_config(layout="wide", page_title="DDoS Attack Prediction System")

st.markdown(
    """
    <style>
    .main {
        background: #1e1e1e;
        color: white;
    }
    .stButton>button {
        background-color: #FF5733;
        color: white;
        font-size: 18px;
        border-radius: 8px;
        width: 100%;
        height: 60px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App header
st.markdown("<h1 style='text-align: center; color: #FF5733;'>❤️ DDoS Attack Prediction System</h1>", unsafe_allow_html=True)

# Start and Stop buttons
col1, col2 = st.columns(2)
start_btn = col1.button("▶️ START", key="start")
stop_btn = col2.button("🛑 STOP", key="stop")

# Simulated traffic generator
def generate_traffic():
    """Simulate network traffic with random normal and DDoS values."""
    if np.random.random() < 0.7:  # 70% normal traffic, 30% DDoS
        return np.array([[np.random.randint(0, 65535),    
                          np.random.randint(100, 10000),  
                          np.random.randint(50, 1500),    
                          np.random.randint(50, 1500),    
                          np.random.uniform(1000, 100000), 
                          np.random.uniform(10, 100),     
                          np.random.uniform(100, 1000)]], 
                        dtype=np.float32)
    else:
        return np.array([[80,              
                          10000,           
                          1400,            
                          1400,            
                          950000.0,        
                          980.0,           
                          5.0]], dtype=np.float32)

# Initialize session state
if 'running' not in st.session_state:
    st.session_state['running'] = False

# Button logic
if start_btn:
    st.session_state['running'] = True

if stop_btn:
    st.session_state['running'] = False

# Detection loop
if st.session_state['running'] and cnn_model and scaler:
    st.write("### 🚀 **Real-Time Network Traffic Detection:**")

    # Visualization placeholders
    gauge_placeholder = st.empty()
    line_chart_placeholder = st.empty()
    traffic_placeholder = st.empty()

    # Dataframe to store real-time traffic
    traffic_df = pd.DataFrame(columns=["Time", "Packets/s", "Bytes/s", "Prediction"])

    # Detection threshold
    threshold = 0.3  

    try:
        while st.session_state['running']:
            # Simulate traffic data
            input_data = generate_traffic()

            # Scale and reshape input
            scaled_input = scaler.transform(input_data)
            model_input_shape = cnn_model.input_shape

            # Reshape based on model input shape
            if len(model_input_shape) == 3:
                scaled_input = scaled_input.reshape(1, 7, 1)
            else:
                scaled_input = scaled_input.reshape(1, 7)

            # Make prediction
            prediction = cnn_model.predict(scaled_input)[0][0]
            is_ddos = prediction > threshold

            # Display traffic values
            traffic_placeholder.markdown(
                f"""
                **Traffic Data:**  
                - 🌐 **Destination Port:** {int(input_data[0][0])}  
                - ⏱️ **Flow Duration:** {int(input_data[0][1])} ms  
                - 📦 **Fwd Packet Length:** {int(input_data[0][2])} bytes  
                - 📦 **Bwd Packet Length:** {int(input_data[0][3])} bytes  
                - 🔥 **Flow Bytes/s:** {input_data[0][4]:,.2f}  
                - 🚀 **Flow Packets/s:** {input_data[0][5]:,.2f}  
                - ⏲️ **Flow IAT Mean:** {input_data[0][6]:,.2f}
                """
            )

            # Append data to the DataFrame
            current_time = time.strftime("%H:%M:%S")
            traffic_df = pd.concat([traffic_df, pd.DataFrame({
                "Time": [current_time],
                "Packets/s": [input_data[0][5]],
                "Bytes/s": [input_data[0][4]],
                "Prediction": ["DDoS" if is_ddos else "Normal"]
            })], ignore_index=True)

            # Keep the last 50 records
            traffic_df = traffic_df.tail(50)

            # Gauge chart
            gauge_value = int(prediction * 100)
            gauge_color = "red" if is_ddos else "green"

            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=gauge_value,
                title={"text": "DDoS Risk Level"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": gauge_color},
                    "steps": [
                        {"range": [0, 30], "color": "green"},
                        {"range": [30, 70], "color": "yellow"},
                        {"range": [70, 100], "color": "red"}
                    ]
                }
            ))

            gauge_placeholder.plotly_chart(gauge_fig, use_container_width=True)

            # Line chart for traffic over time
            line_fig = go.Figure()

            line_fig.add_trace(go.Scatter(
                x=traffic_df["Time"], y=traffic_df["Packets/s"],
                mode='lines+markers', name='Packets/s',
                line=dict(color='lightgreen')
            ))

            line_fig.add_trace(go.Scatter(
                x=traffic_df["Time"], y=traffic_df["Bytes/s"],
                mode='lines+markers', name='Bytes/s',
                line=dict(color='orange')
            ))

            line_chart_placeholder.plotly_chart(line_fig, use_container_width=True)

            # Refresh every 2 seconds
            time.sleep(2)

    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    if not cnn_model or not scaler:
        st.error("Model or Scaler failed to load. Check your file paths.")
    else:
        st.write("Click START to begin real-time detection.")
