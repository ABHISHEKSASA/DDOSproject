import streamlit as st
import numpy as np
import tensorflow as tf
import joblib  # For loading scaler
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model
try:
    model = tf.keras.models.load_model("ddos_cnn_model.h5")
    st.success("✅ Model Loaded Successfully!")
except Exception as e:
    st.error(f"❌ Error Loading Model: {e}")

# Load the pre-trained scaler
try:
    scaler = joblib.load("scaler.pkl")
    st.success("✅ Scaler Loaded Successfully!")
except Exception as e:
    st.error(f"❌ Error Loading Scaler: {e}")

# Define the 17 actual feature names (Ensure these match the training data)
feature_names = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets", 
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max", 
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", 
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean"
]  # ⚠️ Excluding "Label" since it's the target variable

# Streamlit UI
st.title("🚀 DDoS Attack Detection with CNN")
st.write("Enter network traffic data below to check if it's an attack.")

# Create input fields dynamically based on feature names
inputs = {}
for feature in feature_names:
    inputs[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

# Convert input to NumPy array
X_new = np.array([list(inputs.values())]).reshape(1, -1)

# Define prediction function
def predict_ddos(features):
    try:
        features = np.array(features).reshape(1, -1)
        if features.shape[1] != 17:
            return f"❌ Error: Expected 17 features, but received {features.shape[1]}."
        features = scaler.transform(features)  # Scale input
        prediction = model.predict(features)[0][0]
        return "🚨 DDoS Attack Detected!" if prediction > 0.5 else "✅ Benign Traffic"
    except Exception as e:
        return f"❌ Prediction Error: {e}"

# Predict button
if st.button("Predict"):
    result = predict_ddos(list(inputs.values()))
    st.subheader(result)
