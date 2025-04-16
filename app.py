import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import json
import threading
import time
import paho.mqtt.client as mqtt
from math import sqrt, atan2, degrees

# Load model
model = tf.keras.models.load_model("fall_model_bpn.h5")
labels = ['jatuh_samping', 'naik_tangga', 'duduk', 'sujud']
features = ['Ax', 'Ay', 'Az', 'SV', 'Gx', 'Gy', 'Gz', 'Pitch', 'Roll', 'Yaw']

# Shared container
incoming_data = []

# === MQTT CALLBACKS ===
def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT with result code", rc)
    client.subscribe("fall/////")

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        Ax = payload.get("Ax", 0)
        Ay = payload.get("Ay", 0)
        Az = payload.get("Az", 0)
        Gx = payload.get("Gx", 0)
        Gy = payload.get("Gy", 0)
        Gz = payload.get("Gz", 0)

        SV = sqrt(Ax**2 + Ay**2 + Az**2)
        Pitch = degrees(atan2(Ax, sqrt(Ay**2 + Az**2)))
        Roll = degrees(atan2(Ay, sqrt(Ax**2 + Az**2)))
        Yaw = degrees(atan2(Az, sqrt(Ax**2 + Ay**2)))

        row = [Ax, Ay, Az, SV, Gx, Gy, Gz, Pitch, Roll, Yaw]
        incoming_data.append(row)

        print("Received data:", row)  # Optional: Debug MQTT payload

    except Exception as e:
        print("Error processing message:", e)

# === MQTT SETUP ===
def start_mqtt():
    client = mqtt.Client(protocol=mqtt.MQTTv311)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect("broker.emqx.io", 1883, 60)
    client.loop_forever()  # Keep connection alive

# Start MQTT in background
mqtt_thread = threading.Thread(target=start_mqtt, daemon=True)
mqtt_thread.start()

# === STREAMLIT UI ===
st.set_page_config(page_title="Fall Detection", layout="centered")
st.title("📡 Fall Detection (ESP32 via MQTT)")
st.write("Receiving real-time data on topic: `fall`")

status_placeholder = st.empty()
df_placeholder = st.empty()
chart_placeholder = st.empty()

# Auto-prediction loop every 2 seconds
time.sleep(2)

if incoming_data:
    df = pd.DataFrame(incoming_data, columns=features)

    try:
        # Normalize
        scaler = MinMaxScaler()
        X = scaler.fit_transform(df[features].values)

        # Predict
        predictions = model.predict(X)
        predicted_labels = [labels[np.argmax(p)] for p in predictions]
        df['Prediction'] = predicted_labels

        # Display
        status_placeholder.success(f"✅ {len(df)} records received and predicted")
        df_placeholder.dataframe(df.tail(10))
        chart_placeholder.bar_chart(df['Prediction'].value_counts())

        # Clear for next round
        incoming_data.clear()

    except Exception as e:
        status_placeholder.error(f"Prediction error: {e}")
else:
    status_placeholder.info("⏳ Waiting for data from MQTT...")

# Force rerun after delay
time.sleep(2)
st.rerun()
