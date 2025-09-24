import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import time
import os

# Optional Arduino support
try:
    import serial
    ARDUINO_CONNECTED = True
except ImportError:
    ARDUINO_CONNECTED = False

# ---------------- FILE SETTINGS ----------------
ANALYSIS_FILE = "neonatal_incubator_with_actions.xlsx"  # Historical analysis
LIVE_FILE = "neonatal_incubator_data.xlsx"              # Live/simulation data

SIMULATION_INTERVAL = 60  # seconds between simulated readings

# ---------------- THRESHOLDS ----------------
TEMP_LOW, TEMP_HIGH = 36.5, 37.2
HUM_LOW, HUM_HIGH = 50, 65
HR_LOW, HR_HIGH = 120, 160

st.title("🍼 Neonatal Baby Incubator Dashboard")

# ---------------- LOAD DATA ----------------
def load_excel(file):
    if not os.path.exists(file):
        st.error(f"Data file not found: {file}")
        return None
    xls = pd.ExcelFile(file)
    sheet = xls.sheet_names[0]  # First sheet
    df = pd.read_excel(xls, sheet_name=sheet, engine="openpyxl")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

# ---------------- ALERT FUNCTION ----------------
def check_alert(row):
    alerts = []
    if row['temperature'] < TEMP_LOW or row['temperature'] > TEMP_HIGH:
        alerts.append("⚠ Temp")
    if row['humidity'] < HUM_LOW or row['humidity'] > HUM_HIGH:
        alerts.append("⚠ Humidity")
    if row['heart_rate'] < HR_LOW or row['heart_rate'] > HR_HIGH:
        alerts.append("⚠ Heart Rate")
    return ", ".join(alerts) if alerts else "All Normal ✅"

# ---------------- HISTORICAL DATA ----------------
st.subheader("📊 Historical Data Analysis")
df_analysis = load_excel(ANALYSIS_FILE)
if df_analysis is not None:
    df_analysis['alerts'] = df_analysis.apply(check_alert, axis=1)
    
    # Graphs
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df_analysis['timestamp'], df_analysis['temperature'], label='Temperature (°C)', marker='o')
    ax.plot(df_analysis['timestamp'], df_analysis['humidity'], label='Humidity (%)', marker='o')
    ax.plot(df_analysis['timestamp'], df_analysis['weight'], label='Weight (kg)', marker='o')
    ax.plot(df_analysis['timestamp'], df_analysis['heart_rate'], label='Heart Rate (bpm)', marker='o')
    ax.set_xlabel("Time")
    ax.set_ylabel("Values")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Alerts Table
    st.dataframe(df_analysis[['timestamp','temperature','humidity','heart_rate','alerts']])

    # Weight prediction
    df_analysis['time_idx'] = np.arange(len(df_analysis))
    X = df_analysis[['time_idx']]
    y = df_analysis['weight']
    model = LinearRegression()
    model.fit(X, y)
    future_idx = np.arange(len(df_analysis), len(df_analysis)+3).reshape(-1,1)
    pred_weight = model.predict(future_idx)
    st.subheader("📈 Predicted Weight for Next 3 Time Points")
    st.write(pred_weight)

# ---------------- LIVE DATA / SIMULATION ----------------
st.subheader("⏱ Live Data / Simulation")
df_live = load_excel(LIVE_FILE)
if df_live is None:
    df_live = pd.DataFrame(columns=['timestamp','temperature','humidity','weight','heart_rate','alerts'])

live_slot = st.empty()

if ARDUINO_CONNECTED:
    st.info("Arduino detected! Reading live data...")
    # Replace 'COM3' with your Arduino port
    ser = serial.Serial('COM3', 9600, timeout=1)
    while True:
        line = ser.readline().decode().strip()
        if line:
            temp, hum, weight, hr = map(float, line.split(","))
            new_time = pd.Timestamp.now()
            new_row = {
                "timestamp": new_time,
                "temperature": temp,
                "humidity": hum,
                "weight": weight,
                "heart_rate": hr,
                "alerts": check_alert({"temperature":temp,"humidity":hum,"heart_rate":hr})
            }
            df_live = pd.concat([df_live, pd.DataFrame([new_row])], ignore_index=True)
            live_slot.write(df_live.tail(5))
        time.sleep(60)
else:
    st.info("Arduino not detected. Simulating live data every 1 min...")
    for i in range(3):  # simulate 3 new readings
        if df_live.empty:
            last_time = pd.Timestamp.now()
            last_weight = 3.0
        else:
            last = df_live.iloc[-1]
            last_time = last['timestamp']
            last_weight = last['weight']

        new_time = last_time + pd.Timedelta(seconds=SIMULATION_INTERVAL)
        new_row = {
            "timestamp": new_time,
            "temperature": np.random.uniform(36.2,37.5),
            "humidity": np.random.uniform(48,67),
            "weight": last_weight + np.random.uniform(0,0.02),
            "heart_rate": np.random.randint(115,165),
        }
        new_row['alerts'] = check_alert(new_row)
        df_live = pd.concat([df_live, pd.DataFrame([new_row])], ignore_index=True)
        live_slot.write(df_live.tail(5))
        time.sleep(SIMULATION_INTERVAL)
