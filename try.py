import streamlit as st
import pandas as pd
import numpy as np
import json
import threading
import time
import paho.mqtt.client as mqtt
from math import sqrt
import smtplib
from email.mime.text import MIMEText
# Import joblib load.
from joblib import load
import tensorflow as tf  # Import TensorFlow (for loading the model)

# === Streamlit Configuration (MUST BE FIRST) ===
st.set_page_config(page_title="Fall Detection", layout="centered")

# === Model Loading and Feature Extraction (Adapt to your model!) ===
# Load the trained Keras model
try:
    model = tf.keras.models.load_model("fall_detection_model.h5")  # Load the 1D CNN model
    scaler = load('scaler.joblib')  # Load the scaler.
    st.success("✅ Model and Scaler loaded successfully.")
except Exception as e:
    st.error(f"❌ Error loading the model or scaler: {e}.  Make sure the files exist and the paths are correct.")
    st.stop()  # Stop the app if the model fails to load

# Define Features (MUST MATCH what you used during training!)
features = ['AcX', 'AcY', 'AcZ', 'Gx', 'Gy', 'Gz']
window_size_seconds = 1  # MUST MATCH the window_size used when training your model (CHECK TRAINING CODE)
sample_rate = 10  # Match sample_rate from your training code.

# === MQTT Setup ===
# Replace with your MQTT broker details
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC = "fall/////"

# Shared container
incoming_data = []  # Store raw data for the window
sensor_data_buffer = []
fall_detected_flag = False  # Flag to indicate if a fall was detected

# --- Feature Engineering function (MUST MATCH THE FEATURE EXTRACTION IN TRAINING!) ---
def extract_features(window_df):
    """Extracts features from the accelerometer and gyroscope data."""
    # Ensure window_df is not empty and has correct columns
    if window_df.empty or not all(col in window_df.columns for col in ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']):
        print("Warning: Invalid window_df or missing columns")
        return np.zeros(32)  # Return zeros or handle the error appropriately

    # Extract features
    ax = window_df['Ax'].values
    ay = window_df['Ay'].values
    az = window_df['Az'].values
    gx = window_df['Gx'].values
    gy = window_df['Gy'].values
    gz = window_df['Gz'].values

    # Acceleration and Angular velocity magnitudes
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    gyr_mag = np.sqrt(gx**2 + gy**2 + gz**2)

    # Calculate features
    features = [
        np.mean(ax), np.std(ax), np.max(ax), np.min(ax),
        np.mean(ay), np.std(ay), np.max(ay), np.min(ay),
        np.mean(az), np.std(az), np.max(az), np.min(az),
        np.mean(gx), np.std(gx), np.max(gx), np.min(gx),
        np.mean(gy), np.std(gy), np.max(gy), np.min(gy),
        np.mean(gz), np.std(gz), np.max(gz), np.min(gz),
        np.mean(acc_mag), np.std(acc_mag), np.max(acc_mag), np.min(acc_mag),
        np.mean(gyr_mag), np.std(gyr_mag), np.max(gyr_mag), np.min(gyr_mag)
    ]

    return np.array(features)

# --- MQTT Functions ---
def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT with result code " + str(rc))
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    global incoming_data, fall_detected_flag, sensor_data_buffer
    try:
        payload = json.loads(msg.payload.decode())
        #print("Raw Payload:", payload)  # Debug: Print the raw JSON payload

        # Extract sensor readings (handle potential missing keys)
        ax = float(payload.get("Ax", 0.0))  # Default to 0.0 if missing
        ay = float(payload.get("Ay", 0.0))
        az = float(payload.get("Az", 0.0))
        gx = float(payload.get("Gx", 0.0))
        gy = float(payload.get("Gy", 0.0))
        gz = float(payload.get("Gz", 0.0))
        sensor_data_buffer.append([ax, ay, az, gx, gy, gz])  # Store for processing

        # Process the data to detect a fall
        # Create a time window based on sample rate and window size
        window_size = int(window_size_seconds * sample_rate) # must be 50

        if len(sensor_data_buffer) >= window_size:
            # Create a DataFrame for the window
            window_df = pd.DataFrame(sensor_data_buffer[-window_size:], columns=['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz'])
            print("Window Data Shape before feature extraction:", window_df.shape)
            # Feature Engineering
            features_extracted = extract_features(window_df)  # Extract features from the window. Convert to numpy array
            print("Shape of extracted features:", features_extracted.shape)
            # Scale the extracted features.
            features_scaled = scaler.transform(features_extracted.reshape(1, -1))
            print("Shape of scaled features:", features_scaled.shape)
            # Reshape to match the 1D CNN input
            X_reshaped = features_scaled.reshape(1, 32, 1)  # Corrected for 1D CNN input (samples, time_steps, features)
            print(f"Shape of reshaped data: {X_reshaped.shape}")

            # Predict
            prediction = model.predict(X_reshaped)
            print("Prediction", prediction)
            predicted_label = int(round(prediction[0][0]))  # Threshold and get label. round the data.
            #print(f"Prediction: {prediction}")
            fall_detected_flag = (predicted_label == 0) # Set fall_detected_flag based on prediction.
            # The print statement is placed here instead to ensure that the processing occurs at the same time
            # as the call to the detector.
            if fall_detected_flag:
                print("🚨 FALL DETECTED!")
                # You can trigger an action (e.g., send an alert) here

            # Clear the buffer after processing the window.  Only need to save the most recent.
            sensor_data_buffer = sensor_data_buffer[-window_size + 1:] # Keep 1 extra sample.

        # Store the sensor values for display.
        incoming_data.append([ax, ay, az, gx, gy, gz])

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except KeyError as e:
        print(f"Missing key in JSON payload: {e}")
    except Exception as e:
        print(f"Error processing message: {e}")

# --- Initialize MQTT Client ---
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

# --- Start MQTT Connection in a separate thread ---
def mqtt_thread_function():
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()

mqtt_thread = threading.Thread(target=mqtt_thread_function, daemon=True)
mqtt_thread.start()

# --- Streamlit UI (Real-time Display) ---
st.title("📡 Fall Detection (ESP32 via MQTT)")
st.write("Receiving real-time data on topic: `fall`")

# Taking inputs
email_family = st.text_input('Family Member Email')
email_hospital = st.text_input('Hospital Email')

status_placeholder = st.empty()
df_placeholder = st.empty()
chart_placeholder = st.empty()

# --- Main Loop (Runs every 2 seconds) ---
while True:
    time.sleep(2)  # Update every 2 seconds

    if incoming_data:
        try:
            # Display Sensor Readings (Last 5 rows)
            df = pd.DataFrame(incoming_data[-5:], columns=['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz'])  # Show last 5 data readings.
            df_placeholder.dataframe(df)

            # Display fall detection status.
            if fall_detected_flag:
                status_placeholder.error("🚨 FALL DETECTED!")
                try:
                    body =  "Halo,Keluarga Anda ada Yang Terjatuh. Segera Periksa!"
                    msg = MIMEText(body)
                    msg['From'] = st.secrets["email"]
                    msg['To'] = email_family 
                    msg['Cc'] = email_hospital  # Add hospital email to CC
                    msg['Subject'] = "Alert: Fall Detected!"

                    server = smtplib.SMTP(st.secrets["smtp-address"], 587)
                    server.starttls()
                    server.login(st.secrets["email"], st.secrets["password"])
                    server.sendmail(st.secrets["email"], email_family, msg.as_string())
                    server.sendmail(st.secrets["email"], email_hospital, msg.as_string())
                    server.quit()

                    st.success('Email sent successfully! 🚀')
                except Exception as e:
                    st.error(f"Error sending email: {e}")
                fall_detected_flag = False  # Reset the flag.
            else:
                status_placeholder.info("✅ Monitoring...")

            # Clear data after processing
            incoming_data = incoming_data[-5:]  # Keep the last 5 readings for display

        except Exception as e:
            status_placeholder.error(f"UI Update Error: {e}")
    else:
        status_placeholder.info("⏳ Waiting for data from MQTT...")

# Force rerun after delay
time.sleep(2)
st.rerun()