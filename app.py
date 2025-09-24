import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import time

# Optional: Arduino import
try:
    import serial
    ARDUINO_CONNECTED = True
except:
    ARDUINO_CONNECTED = False

# ---------------- SETTINGS ----------------
DATA_FILE = "neonatal_incubator_data.xlsx"   # fallback file
SIMULATION_INTERVAL = 60  # seconds between readings

TEMP_LOW, TEMP_HIGH = 36.5, 37.2
HUM_LOW, HUM_HIGH = 50, 65
HR_LOW, HR_HIGH = 120, 160

st.title("🍼 Neonatal Baby Incubator Dashboard")

# ---------------- LOAD DATA ----------------
def load_data():
    xls = pd.ExcelFile(DATA_FILE)
    sheet = xls.sheet_names[0]  # first sheet
    df = pd.read_excel(xls, sheet_name=sheet, engine="openpyxl")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

df = load_data()

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

df['alerts'] = df.apply(check_alert, axis=1)

# ---------------- GRAPHS ----------------
st.subheader("📊 Parameter Trends")
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(df['timestamp'], df['temperature'], label='Temperature (°C)', marker='o')
ax.plot(df['timestamp'], df['humidity'], label='Humidity (%)', marker='o')
ax.plot(df['timestamp'], df['weight'], label='Weight (kg)', marker='o')
ax.plot(df['timestamp'], df['heart_rate'], label='Heart Rate (bpm)', marker='o')
ax.set_xlabel("Time")
ax.set_ylabel("Values")
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.subheader("⚠ Alerts Overview")
st.dataframe(df[['timestamp','temperature','humidity','heart_rate','alerts']])

# ---------------- PREDICTION ----------------
st.subheader("📈 Weight Prediction (Next 3 readings)")
df['time_idx'] = np.arange(len(df))
X = df[['time_idx']]
y = df['weight']
model = LinearRegression()
model.fit(X, y)
future_idx = np.arange(len(df), len(df)+3).reshape(-1,1)
pred_weight = model.predict(future_idx)
st.write(pred_weight)

# ---------------- LIVE SIMULATION / ARDUINO ----------------
st.subheader("⏱ Live Data Simulation or Arduino Readings")
live_slot = st.empty()

if ARDUINO_CONNECTED:
    st.info("Arduino detected! Reading live data...")
    # Example: change 'COM3' to your Arduino port and 9600 to your baud rate
    ser = serial.Serial('COM3', 9600, timeout=1)
    for _ in range(5):  # read 5 live rows for demo
        line = ser.readline().decode().strip()  # Arduino should send CSV: temp,hum,weight,hr
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
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            live_slot.write(df.tail(5))
            time.sleep(60)
else:
    st.info("Arduino not detected. Simulating live data every 1 min...")
    for i in range(3):
        last = df.iloc[-1]
        new_time = last['timestamp'] + pd.Timedelta(seconds=SIMULATION_INTERVAL)
        new_row = {
            "timestamp": new_time,
            "temperature": np.random.uniform(36.2,37.5),
            "humidity": np.random.uniform(48,67),
            "weight": last['weight'] + np.random.uniform(0,0.02),
            "heart_rate": np.random.randint(115,165),
            "alerts": ""
        }
        new_row['alerts'] = check_alert(new_row)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        live_slot.write(df.tail(5))
        time.sleep(SIMULATION_INTERVAL)
