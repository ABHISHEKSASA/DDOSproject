import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

# Streamlit Title
st.title("DDoS Attack Detection (CNN)")

# File Uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        # Read the CSV File
        df = pd.read_csv(uploaded_file)
        st.write("✅ Dataset Loaded Successfully!")

        # Check if 'Label' column exists
        if 'Label' not in df.columns:
            st.error("❌ No 'Label' column found. Make sure your dataset has a 'Label' column.")
        else:
            # Convert Label column to binary (if needed)
            df['Label'] = df['Label'].map({'Benign': 0, 'DDoS': 1})

            # Prepare Features and Labels
            X = df.drop(columns=['Label'])
            y = df['Label']

            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # CNN Model Training Function
            def train_cnn(X_train, y_train, X_test, y_test):
                X_train_reshaped = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
                X_test_reshaped = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

                model = keras.Sequential([
                    keras.layers.Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
                    keras.layers.MaxPooling1D(2),
                    keras.layers.Flatten(),
                    keras.layers.Dense(64, activation='relu'),
                    keras.layers.Dense(1, activation='sigmoid')
                ])

                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                model.fit(X_train_reshaped, y_train, epochs=10, validation_data=(X_test_reshaped, y_test), verbose=0)
                return model

            # Train the CNN Model
            st.write("Training CNN Model... ⏳")
            cnn_model = train_cnn(X_train, y_train, X_test, y_test)
            st.write("✅ Model Training Complete!")

            # Sidebar for Feature Selection
            st.sidebar.header("Feature Selection")
            selected_features = st.sidebar.multiselect("Select Features for Prediction", options=X.columns, default=X.columns[:5])

            # Show Input Fields for Selected Features
            st.sidebar.header("Input Values")
            input_features = {col: st.sidebar.number_input(f"{col}", value=float(X[col].mean())) for col in selected_features}

            # Prediction Button
            if st.button("Predict"):
                if not selected_features:
                    st.error("❌ Please select at least one feature for prediction.")
                else:
                    input_df = pd.DataFrame([input_features])
                    input_reshaped = input_df.values.reshape(1, input_df.shape[1], 1)
                    prediction = (cnn_model.predict(input_reshaped) > 0.5).astype("int32")

                    # Display Prediction
                    st.error("🚨 DDoS Attack Detected!") if prediction[0] == 1 else st.success("✅ Benign Traffic Detected!")

    except Exception as e:
        st.error(f"Error reading file: {e}")
