import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LinearRegression
import time
import os

# ---------------- FILE SETTINGS ----------------
HISTORICAL_FILE = "neonatal_incubator_with_actions.xlsx"
LIVE_FILE = "neonatal_incubator_data.xlsx"
SIMULATION_INTERVAL = 60  # seconds for live simulation

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
    df = pd.read_excel(file, engine="openpyxl")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

# ---------------- ALERT FUNCTION ----------------
def check_alert(row):
    alerts = []
    if row['temperature'] < TEMP_LOW or row['temperature'] > TEMP_HIGH:
        alerts.append("Temperature")
    if row['humidity'] < HUM_LOW or row['humidity'] > HUM_HIGH:
        alerts.append("Humidity")
    if row['heart_rate'] < HR_LOW or row['heart_rate'] > HR_HIGH:
        alerts.append("Heart Rate")
    return alerts

# ---------------- MODE SELECTION ----------------
mode = st.sidebar.radio("Select Mode:", ["Historical Analysis", "Live / Simulation"])

# ---------------- HISTORICAL ANALYSIS ----------------
if mode == "Historical Analysis":
    st.subheader("📊 Historical Data Analysis")
    df = load_excel(HISTORICAL_FILE)
    if df is not None:
        df['alerts'] = df.apply(check_alert, axis=1)
        df['alerts_text'] = df['alerts'].apply(lambda x: ", ".join(x) if x else "Normal ✅")

        # Interactive charts
        for param in ['temperature','humidity','weight','heart_rate']:
            fig = px.line(df, x='timestamp', y=param, title=f"{param.capitalize()} Trend")
            alerts_df = df[df['alerts'].apply(lambda x: param in x)]
            if not alerts_df.empty:
                fig.add_scatter(x=alerts_df['timestamp'], y=alerts_df[param],
                                mode='markers', marker=dict(color='red', size=10),
                                name='Alert')
            st.plotly_chart(fig, use_container_width=True)

        # Latest alerts
        latest = df.iloc[-1]
        if latest['alerts']:
            st.warning(f"⚠ Emergency Alert: {', '.join(latest['alerts'])} at {latest['timestamp']}")
        else:
            st.success("✅ All parameters normal")

        # Optional: weight growth prediction
        df['time_idx'] = np.arange(len(df))
        X = df[['time_idx']]
        y = df['weight']
        model = LinearRegression()
        model.fit(X, y)
        future_idx = np.arange(len(df), len(df)+3).reshape(-1,1)
        pred_weight = model.predict(future_idx)
        st.subheader("📈 Predicted Weight for Next 3 Time Points")
        st.write(pred_weight)

# ---------------- LIVE / SIMULATION ----------------
elif mode == "Live / Simulation":
    st.subheader("⏱ Live Data / Simulation")
    df_live = load_excel(LIVE_FILE)
    if df_live is None:
        df_live = pd.DataFrame(columns=['timestamp','temperature','humidity','weight','heart_rate','alerts'])

    live_slot = st.empty()
    chart_slots = {param: st.empty() for param in ['temperature','humidity','weight','heart_rate']}

    def add_new_row(temp, hum, weight, hr):
        new_time = pd.Timestamp.now()
        row_series = pd.Series({"temperature": temp, "humidity": hum, "weight": weight, "heart_rate": hr})
        alerts = check_alert(row_series)
        new_row = pd.DataFrame([{
            "timestamp": new_time,
            "temperature": temp,
            "humidity": hum,
            "weight": weight,
            "heart_rate": hr,
            "alerts": alerts
        }])
        return new_row

    st.info("Simulating live data. App refreshes every minute automatically.")

    for i in range(3):  # simulate 3 live readings
        if df_live.empty:
            last_weight = 3.0
        else:
            last_weight = df_live.iloc[-1]['weight']

        temp = np.random.uniform(36.2, 37.5)
        hum = np.random.uniform(48, 67)
        weight = last_weight + np.random.uniform(0, 0.02)
        hr = np.random.randint(115, 165)

        new_df = add_new_row(temp, hum, weight, hr)
        df_live = pd.concat([df_live, new_df], ignore_index=True)

        # Update interactive charts
        for param in ['temperature','humidity','weight','heart_rate']:
            fig = px.line(df_live, x='timestamp', y=param, title=f"{param.capitalize()} Trend")
            alerts_df = df_live[df_live['alerts'].apply(lambda x: param in x)]
            if not alerts_df.empty:
                fig.add_scatter(x=alerts_df['timestamp'], y=alerts_df[param],
                                mode='markers', marker=dict(color='red', size=10),
                                name='Alert')
            chart_slots[param].plotly_chart(fig, use_container_width=True)

        # Show latest 5 readings
        df_live['alerts_text'] = df_live['alerts'].apply(lambda x: ", ".join(x) if x else "Normal ✅")
        live_slot.dataframe(df_live.tail(5))

        # Emergency alert
        latest = df_live.iloc[-1]
        if latest['alerts']:
            st.warning(f"⚠ Emergency Alert: {', '.join(latest['alerts'])} at {latest['timestamp']}")
        else:
            st.success("✅ All parameters normal")

        time.sleep(SIMULATION_INTERVAL)



"""
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
ANALYSIS_FILE = "neonatal_incubator_with_actions.xlsx"
LIVE_FILE = "neonatal_incubator_data.xlsx"
SIMULATION_INTERVAL = 60  # seconds

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
    sheet = xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sheet, engine="openpyxl")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

# ---------------- ALERT FUNCTION ----------------
def check_alert(row):
    alerts = []
    if row['temperature'] < TEMP_LOW or row['temperature'] > TEMP_HIGH:
        alerts.append("Temperature")
    if row['humidity'] < HUM_LOW or row['humidity'] > HUM_HIGH:
        alerts.append("Humidity")
    if row['heart_rate'] < HR_LOW or row['heart_rate'] > HR_HIGH:
        alerts.append("Heart Rate")
    return alerts

# ---------------- HISTORICAL DATA ----------------
st.sidebar.subheader("Mode Selection")
mode = st.sidebar.radio("Select Mode:", ["Historical Analysis", "Live / Simulation"])

if mode == "Historical Analysis":
    st.subheader("📊 Historical Data Analysis")
    df_analysis = load_excel(ANALYSIS_FILE)
    if df_analysis is not None:
        df_analysis['alerts'] = df_analysis.apply(check_alert, axis=1)
        
        # Individual Graphs
        parameters = ['temperature','humidity','weight','heart_rate']
        for param in parameters:
            st.write(f"### {param.capitalize()} Trend")
            fig, ax = plt.subplots(figsize=(12,4))
            ax.plot(df_analysis['timestamp'], df_analysis[param], marker='o', color='blue')
            # Highlight alerts
            for i, row in df_analysis.iterrows():
                if param in row['alerts']:
                    ax.plot(row['timestamp'], row[param], marker='o', color='red', markersize=8)
            ax.set_xlabel("Time")
            ax.set_ylabel(param.capitalize())
            ax.grid(True)
            st.pyplot(fig)

        # Show alerts table
        st.subheader("Alerts Table")
        df_analysis['alerts_text'] = df_analysis['alerts'].apply(lambda x: ", ".join(x) if x else "Normal ✅")
        st.dataframe(df_analysis[['timestamp','temperature','humidity','heart_rate','weight','alerts_text']])

        # Weight Prediction
        df_analysis['time_idx'] = np.arange(len(df_analysis))
        X = df_analysis[['time_idx']]
        y = df_analysis['weight']
        model = LinearRegression()
        model.fit(X, y)
        future_idx = np.arange(len(df_analysis), len(df_analysis)+3).reshape(-1,1)
        pred_weight = model.predict(future_idx)
        st.subheader("📈 Predicted Weight for Next 3 Time Points")
        st.write(pred_weight)

# ---------------- LIVE / SIMULATION ----------------
elif mode == "Live / Simulation":
    st.subheader("⏱ Live Data / Simulation")
    df_live = load_excel(LIVE_FILE)
    if df_live is None:
        df_live = pd.DataFrame(columns=['timestamp','temperature','humidity','weight','heart_rate','alerts'])

    live_slot = st.empty()
    chart_slots = {
        'temperature': st.empty(),
        'humidity': st.empty(),
        'weight': st.empty(),
        'heart_rate': st.empty()
    }

    def add_new_row(temp, hum, weight, hr):
        new_time = pd.Timestamp.now()
        new_row = {
            "timestamp": new_time,
            "temperature": temp,
            "humidity": hum,
            "weight": weight,
            "heart_rate": hr,
            "alerts": check_alert({"temperature":temp,"humidity":hum,"heart_rate":hr})
        }
        return pd.DataFrame([new_row])

    # Simulate live readings every 1 min
    for i in range(3):  # 3 simulated readings for demonstration
        if df_live.empty:
            last_weight = 3.0
        else:
            last_weight = df_live.iloc[-1]['weight']

        # Generate simulated data
        temp = np.random.uniform(36.2,37.5)
        hum = np.random.uniform(48,67)
        weight = last_weight + np.random.uniform(0,0.02)
        hr = np.random.randint(115,165)

        new_df = add_new_row(temp, hum, weight, hr)
        df_live = pd.concat([df_live, new_df], ignore_index=True)

        # Update charts individually
        for param in ['temperature','humidity','weight','heart_rate']:
            fig, ax = plt.subplots(figsize=(10,3))
            ax.plot(df_live['timestamp'], df_live[param], marker='o', color='green')
            # highlight alerts
            for j, row in df_live.iterrows():
                if param in row['alerts']:
                    ax.plot(row['timestamp'], row[param], marker='o', color='red', markersize=8)
            ax.set_xlabel("Time")
            ax.set_ylabel(param.capitalize())
            ax.grid(True)
            chart_slots[param].pyplot(fig)

        # Update table
        df_live['alerts_text'] = df_live['alerts'].apply(lambda x: ", ".join(x) if x else "Normal ✅")
        live_slot.dataframe(df_live.tail(5))

        # Alert message if emergency
        latest_alerts = df_live.iloc[-1]['alerts']
        if latest_alerts:
            st.warning(f"⚠ Emergency Alert: {', '.join(latest_alerts)} at {df_live.iloc[-1]['timestamp']}")
        else:
            st.success("✅ All parameters normal")

        time.sleep(SIMULATION_INTERVAL)
        """
