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

latest_values = {
    "ax": 0,
    "ay": 0,
    "az": 0,
    "gx": 0,
    "gy": 0,
    "gz": 0,
}

# === MQTT CALLBACKS ===
def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT with result code", rc)
    client.subscribe("/v1.6/devices/esp32-uni159/ax/lv")
    client.subscribe("/v1.6/devices/esp32-uni159/ay/lv")
    client.subscribe("/v1.6/devices/esp32-uni159/az/lv")
    client.subscribe("/v1.6/devices/esp32-uni159/gx/lv")
    client.subscribe("/v1.6/devices/esp32-uni159/gy/lv")
    client.subscribe("/v1.6/devices/esp32-uni159/gz/lv")

def on_message(client, userdata, msg):
    try:
        topic = msg.topic.split("/")[-2]
        value = float(msg.payload.decode())

        if topic in latest_values:
            latest_values[topic] = value

        ax = latest_values["ax"]
        ay = latest_values["ay"]
        az = latest_values["az"]
        gx = latest_values["gx"]
        gy = latest_values["gy"]
        gz = latest_values["gz"]

        SV = sqrt(ax**2 + ay**2 + az**2)
        Pitch = degrees(atan2(ax, sqrt(ay**2 + az**2)))
        Roll = degrees(atan2(ay, sqrt(ax**2 + az**2)))
        Yaw = degrees(atan2(az, sqrt(ax**2 + ay**2)))

        row = [ax, ay, az, SV, gx, gy, gz, Pitch, Roll, Yaw]
        incoming_data.append(row)

        print("Received data:", row)

    except Exception as e:
        print("Error processing message:", e)


# === MQTT SETUP ===
def start_mqtt():
    client = mqtt.Client(protocol=mqtt.MQTTv311,)
    client.on_connect = on_connect
    client.on_message = on_message
    client.username_pw_set("BBUS-6L3O4cSRsFqpUEW4e5dJNVxCyiOwJa")
    client.connect("industrial.api.ubidots.com", 1883, 60)
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
