import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

# ===========================
# ✅ Model and Data Functions
# ===========================

@st.cache_resource
def create_model():
    """Create and compile the LSTM model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(7, 1)),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


@st.cache_resource
def generate_data():
    """Generate dummy DDoS and normal traffic data."""
    np.random.seed(42)

    # Simulate normal traffic
    normal_data = np.random.normal(loc=500, scale=100, size=(1000, 7))
    normal_labels = np.zeros((1000, 1))

    # Simulate DDoS traffic
    ddos_data = np.random.normal(loc=10000, scale=2000, size=(500, 7))
    ddos_labels = np.ones((500, 1))

    # Combine and shuffle
    X = np.vstack((normal_data, ddos_data))
    y = np.vstack((normal_labels, ddos_labels))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape for LSTM
    X_train_scaled = X_train_scaled.reshape(-1, 7, 1)
    X_test_scaled = X_test_scaled.reshape(-1, 7, 1)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


@st.cache_resource(show_spinner=True)
def train_model(cache_key):
    """Train the LSTM model."""
    model = create_model()
    X_train, X_test, y_train, y_test, scaler = generate_data()

    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=0)

    # Precision-recall curve
    predictions = model.predict(X_test).flatten()
    precision, recall, thresholds = precision_recall_curve(y_test.flatten(), predictions)

    # Handle threshold calculation properly
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-10)  # Prevent divide by zero
    best_threshold = thresholds[np.argmax(f1_scores) - 1] if len(thresholds) < len(f1_scores) else thresholds[np.argmax(f1_scores)]

    return model, scaler, best_threshold


# ===========================
# ✅ UI and User Input
# ===========================

st.title("🚀 DDoS Attack Detection System")
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

# ===========================
# ✅ Model Prediction
# ===========================

if st.button("🚀 Predict DDoS Attack"):
    model, scaler, best_threshold = train_model(np.random.randint(0, 10000))

    # Scale input data
    scaled_input_data = scaler.transform(input_data)
    scaled_input_data = scaled_input_data.reshape(1, 7, 1)

    # Make prediction
    prediction = model.predict(scaled_input_data)[0][0]

    # Display results
    st.write(f"### ⚙️ Raw Prediction Probability: {prediction:.4f}")
    st.write(f"### 🚦 Threshold: {best_threshold:.4f}")

    if prediction > best_threshold:
        st.error("🚀 **DDoS Attack Detected!**")
    else:
        st.success("✅ **Normal Traffic**")
