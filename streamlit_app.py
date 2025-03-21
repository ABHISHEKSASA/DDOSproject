import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# ==============================
# ✅ Model and Scaler Definition
# ==============================

@st.cache_resource
def create_and_train_model():
    """Creates and trains a simple CNN model for demonstration."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(7, 1)),
        tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification (DDoS or Normal)
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Dummy training for demonstration purposes
    X_train = np.random.rand(100, 7, 1)
    y_train = np.random.randint(0, 2, 100)

    model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=0)
    
    st.success("✅ Model Created and Trained!")
    return model

@st.cache_resource
def create_scaler():
    """Creates and fits a scaler using random training data."""
    X_train = np.random.rand(100, 7)
    scaler = StandardScaler()
    scaler.fit(X_train)
    st.success("✅ Scaler Created and Fitted!")
    return scaler

# Load model and scaler
cnn_model = create_and_train_model()
scaler = create_scaler()

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
        scaled_input_data = scaled_input_data.reshape(1, 7, 1)

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
