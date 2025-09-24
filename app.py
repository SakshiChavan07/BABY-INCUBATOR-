# neonatal_dashboard_final_fixed.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import timedelta
from sklearn.linear_model import LinearRegression
import os
import time

# ---------------- Settings ----------------
HISTORICAL_FILE = "neonatal_incubator_with_actions.xlsx"
LIVE_FILE = "neonatal_live.csv"
REFRESH_SECONDS = 120  # Auto-refresh every 2 minutes
LAST_N = 20  # Last 20 readings

# Safe thresholds
TEMP_LOW, TEMP_HIGH = 36.5, 37.2
HUM_LOW, HUM_HIGH = 50, 65
HR_LOW, HR_HIGH = 120, 160

# ---------------- Page Config ----------------
st.set_page_config(page_title="🍼 Neonatal Incubator Dashboard", layout="wide")
st.title("🍼 Neonatal Baby Incubator Dashboard")

# ---------------- Mode Selection ----------------
mode = st.sidebar.radio("Select Mode:", ["Historical", "Live"])
st.sidebar.markdown("Choose Historical to see past data or Live to connect Arduino (simulated if not connected).")

# ---------------- Helper Functions ----------------
def check_alerts(row):
    alerts = []
    if row['temperature'] < TEMP_LOW or row['temperature'] > TEMP_HIGH:
        alerts.append("Temperature")
    if row['humidity'] < HUM_LOW or row['humidity'] > HUM_HIGH:
        alerts.append("Humidity")
    if row['heart_rate'] < HR_LOW or row['heart_rate'] > HR_HIGH:
        alerts.append("Heart Rate")
    return alerts

def load_excel(file):
    if not os.path.exists(file):
        st.warning(f"File not found: {file}")
        return pd.DataFrame()
    df = pd.read_excel(file, engine="openpyxl")
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['alerts'] = df.apply(check_alerts, axis=1)
    return df

def load_csv(file):
    if not os.path.exists(file):
        return pd.DataFrame()
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['alerts'] = df.apply(check_alerts, axis=1)
    return df

def predict_next(df, param):
    X = np.arange(len(df)).reshape(-1,1)
    y = df[param].values
    model = LinearRegression()
    model.fit(X, y)
    future_idx = np.array([[len(df)], [len(df)+1], [len(df)+2]])
    pred = model.predict(future_idx)
    # Check predicted alerts
    if param=='temperature':
        limits = (TEMP_LOW, TEMP_HIGH)
    elif param=='humidity':
        limits = (HUM_LOW, HUM_HIGH)
    elif param=='heart_rate':
        limits = (HR_LOW, HR_HIGH)
    predicted_alerts = [val for val in pred if val < limits[0] or val > limits[1]]
    return pred, predicted_alerts

def predict_weight(df):
    X = np.arange(len(df)).reshape(-1,1)
    y = df['weight'].values
    model = LinearRegression()
    model.fit(X, y)
    future_idx = np.array([[len(df)], [len(df)+1], [len(df)+2]])
    pred = model.predict(future_idx)
    return pred

# ---------------- Load Data ----------------
if mode == "Historical":
    df = load_excel(HISTORICAL_FILE)
    if df.empty:
        st.stop()
else:
    df = load_csv(LIVE_FILE)
    if df.empty:
        st.info("⚠ Please connect Arduino / ESP32 to fetch live data. Currently showing simulated or last saved data.")
        df = load_excel(HISTORICAL_FILE)
        if df.empty:
            st.stop()

# ---------------- Latest Reading ----------------
latest = df.iloc[-1]
st.subheader("Latest Reading")
st.markdown(f"**Temperature:** {latest.temperature:.2f} °C")
st.markdown(f"**Humidity:** {latest.humidity:.1f} %")
st.markdown(f"**Weight:** {latest.weight:.3f} kg")
st.markdown(f"**Heart Rate:** {int(latest.heart_rate)} bpm")

# ---------------- Emergency Alerts ----------------
alerts_text = [f"{a} at {latest['timestamp']}" for a in latest['alerts']]
if alerts_text:
    st.markdown(f"<p style='color:red; font-weight:bold;'>⚠ Emergency Alerts: {', '.join(alerts_text)}</p>", unsafe_allow_html=True)
else:
    st.success("✅ All parameters normal")

# ---------------- Parameter Graphs ----------------
st.subheader(f"Parameter Graphs (Last {LAST_N} readings)")
last_readings = df.tail(LAST_N)
for param in ['temperature','humidity','weight','heart_rate']:
    fig = px.line(last_readings, x='timestamp', y=param, markers=True, title=param.capitalize())
    alert_mask = last_readings['alerts'].apply(lambda x: param in x)
    if alert_mask.any():
        fig.add_scatter(
            x=last_readings['timestamp'][alert_mask],
            y=last_readings[param][alert_mask],
            mode='markers',
            marker=dict(color='red', size=12),
            name='Alert'
        )
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Predictions ----------------
st.subheader("Predictions for Next Intervals")
for param in ['temperature','humidity','heart_rate']:
    pred_vals, pred_alerts = predict_next(last_readings, param)
    st.write(f"{param.capitalize()} predicted for next 3 intervals: {np.round(pred_vals,2)}")
    if pred_alerts:
        st.markdown(f"<p style='color:red;'>⚠ Predicted {param} Alert: {np.round(pred_alerts,2)}</p>", unsafe_allow_html=True)

# Weight growth
weight_pred = predict_weight(last_readings)
st.write(f"Predicted Weight for next 3 intervals: {np.round(weight_pred,3)} kg")

# ---------------- Baby Growth Report ----------------
st.subheader("📋 Baby Growth Report")
weight_change = last_readings['weight'].iloc[-1] - last_readings['weight'].iloc[0]
st.write(f"Weight change in last {LAST_N} readings: {weight_change:.3f} kg")
if weight_change > 0:
    st.success("✅ Baby is gaining weight")
else:
    st.warning("⚠ Weight gain is low or negative. Monitor closely.")

# ---------------- Doctor Recommendation ----------------
st.subheader("💡 Recommendation for Doctor")
doctor_msgs = []
if any([predict_next(last_readings, p)[1] for p in ['temperature','humidity','heart_rate']]):
    doctor_msgs.append("⚠ Predicted parameter(s) may go out of safe range soon. Monitor closely.")
if weight_change <= 0:
    doctor_msgs.append("⚠ Weight gain is insufficient. Consider nutritional intervention.")
if not doctor_msgs:
    doctor_msgs.append("✅ All parameters within normal limits. Continue regular monitoring.")
for msg in doctor_msgs:
    st.markdown(msg)

# ---------------- Auto-refresh ----------------
st.markdown(f"App auto-refreshes every {REFRESH_SECONDS} seconds.")
time.sleep(REFRESH_SECONDS)
st.experimental_rerun()


"""
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from numpy import polyfit, polyval
import os
import time

st.set_page_config(page_title="🍼 Neonatal Incubator Dashboard", layout="wide")

# ----- Settings -----
HISTORICAL_EXCEL = "neonatal_incubator_with_actions.xlsx"
LIVE_CSV = "neonatal_incubator_data.csv"  # from Arduino/ESP32
REFRESH_SECONDS = 120  # refresh every 2 minutes
PREDICT_MINUTES = 10

# Thresholds
TEMP_LOW, TEMP_HIGH = 36.5, 37.2
HUM_LOW, HUM_HIGH = 50, 65
HR_LOW, HR_HIGH = 120, 160

# ----- Load historical data -----
def load_historical():
    if not os.path.exists(HISTORICAL_EXCEL):
        st.warning("Historical Excel file not found.")
        return pd.DataFrame()
    xls = pd.ExcelFile(HISTORICAL_EXCEL, engine='openpyxl')
    sheet = xls.sheet_names[0]  # automatically take first sheet
    df = pd.read_excel(xls, sheet_name=sheet, engine='openpyxl')
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    else:
        df['timestamp'] = pd.Timestamp.now()
    return df.sort_values('timestamp').reset_index(drop=True)

# ----- Load live data -----
def load_live():
    if os.path.exists(LIVE_CSV):
        df = pd.read_csv(LIVE_CSV)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        else:
            df['timestamp'] = pd.Timestamp.now()
        for col in ['fan_status','heater_status','alarm_status']:
            if col not in df.columns:
                df[col] = 0
        return df.sort_values('timestamp').reset_index(drop=True)
    else:
        return pd.DataFrame()

# ----- Alert function -----
def check_alerts(row):
    alerts = []
    if row['temperature'] < TEMP_LOW or row['temperature'] > TEMP_HIGH:
        alerts.append("Temperature")
    if row['humidity'] < HUM_LOW or row['humidity'] > HUM_HIGH:
        alerts.append("Humidity")
    if row['heart_rate'] < HR_LOW or row['heart_rate'] > HR_HIGH:
        alerts.append("Heart Rate")
    return alerts

# ----- Sidebar mode selection -----
st.sidebar.subheader("Mode Selection")
mode = st.sidebar.radio("Choose mode:", ["Historical", "Live"])

if mode == "Historical":
    st.subheader("📊 Historical Data Analysis")
    df_hist = load_historical()
    if df_hist.empty:
        st.warning("No historical data available.")
        st.stop()

    # Compute alerts
    df_hist['alerts'] = df_hist.apply(check_alerts, axis=1)
    df_hist['alerts_text'] = df_hist['alerts'].apply(lambda x: ", ".join(x) if x else "Normal ✅")

    # Individual graphs
    for param in ['temperature','humidity','weight','heart_rate']:
        st.write(f"### {param.capitalize()} Trend")
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(df_hist['timestamp'], df_hist[param], marker='o', color='blue')
        for i, row in df_hist.iterrows():
            if param in row['alerts']:
                ax.plot(row['timestamp'], row[param], marker='o', color='red', markersize=8)
        ax.set_xlabel("Time")
        ax.set_ylabel(param.capitalize())
        ax.grid(True)
        st.pyplot(fig)

    # Alerts table
    st.subheader("Alerts Table")
    st.dataframe(df_hist[['timestamp','temperature','humidity','heart_rate','weight','alerts_text']])

    # Simple prediction (weight and HR)
    if len(df_hist) >= 5:
        df_hist['time_idx'] = np.arange(len(df_hist))
        future_idx = np.arange(df_hist['time_idx'].iloc[-1]+1, df_hist['time_idx'].iloc[-1]+1+PREDICT_MINUTES)
        pred_weight = polyval(polyfit(df_hist['time_idx'], df_hist['weight'], 1), future_idx)
        pred_hr = polyval(polyfit(df_hist['time_idx'], df_hist['heart_rate'], 1), future_idx)
        future_times = [df_hist['timestamp'].iloc[-1] + timedelta(minutes=i+1) for i in range(PREDICT_MINUTES)]
        pred_df = pd.DataFrame({
            'timestamp': future_times,
            'predicted_weight': np.round(pred_weight,3),
            'predicted_heart_rate': np.round(pred_hr,1)
        })
        st.subheader("📈 Simple Forecast (Next Minutes)")
        st.dataframe(pred_df)
    else:
        st.info("Not enough data for prediction (need >=5 readings).")

elif mode == "Live":
    st.subheader("⏱ Live Data")
    df_live = load_live()
    if df_live.empty:
        st.warning("No live data found. Make sure Arduino/ESP32 is running and CSV is updated.")
        st.stop()

    # Compute alerts
    df_live['alerts'] = df_live.apply(check_alerts, axis=1)
    df_live['alerts_text'] = df_live['alerts'].apply(lambda x: ", ".join(x) if x else "Normal ✅")

    # Latest reading
    latest = df_live.iloc[-1]
    st.subheader("Latest Reading")
    st.metric("Temperature (°C)", f"{latest.temperature:.2f}")
    st.metric("Humidity (%)", f"{latest.humidity:.1f}")
    st.metric("Weight (kg)", f"{latest.weight:.3f}")
    st.metric("Heart Rate (bpm)", f"{int(latest.heart_rate)}")

    # Device actions
    st.subheader("Device Actions (Latest)")
    cols = st.columns(3)
    cols[0].markdown(f"**Fan**: {'🔴 ON' if int(latest.get('fan_status',0))==1 else '🟢 OFF'}")
    cols[1].markdown(f"**Heater**: {'🔴 ON' if int(latest.get('heater_status',0))==1 else '🟢 OFF'}")
    cols[2].markdown(f"**Alarm**: {'🔴 ACTIVE' if int(latest.get('alarm_status',0))==1 else '🟢 OK'}")

    # Graphs (last 300 points)
    st.subheader("Parameter Graphs")
    to_plot = df_live.set_index('timestamp')[['temperature','humidity','weight','heart_rate']].tail(300)
    st.line_chart(to_plot['temperature'].rename("Temperature (°C)").to_frame())
    st.line_chart(to_plot['humidity'].rename("Humidity (%)").to_frame())
    st.line_chart(to_plot['heart_rate'].rename("Heart Rate (bpm)").to_frame())
    st.line_chart(to_plot['weight'].rename("Weight (kg)").to_frame())

    # Alerts table (last 20)
    st.subheader("Recent Alerts")
    alerts = df_live[(df_live['temperature'] < TEMP_LOW) | (df_live['temperature'] > TEMP_HIGH) |
                     (df_live['humidity'] < HUM_LOW) | (df_live['humidity'] > HUM_HIGH) |
                     (df_live['heart_rate'] < HR_LOW) | (df_live['heart_rate'] > HR_HIGH)]
    st.dataframe(alerts[['timestamp','temperature','humidity','heart_rate']].tail(20))

    # Emergency message
    latest_alerts = latest['alerts']
    if latest_alerts:
        st.error(f"⚠ Emergency Alert: {', '.join(latest_alerts)} at {latest['timestamp']}")
    else:
        st.success("✅ All parameters normal")

# Auto-refresh info
st.markdown(f"App updates every **{REFRESH_SECONDS} seconds**.")
st.experimental_rerun()

"""
"""
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from numpy import polyfit, polyval
import os
import time

st.set_page_config(page_title="🍼 Neonatal Incubator Dashboard", layout="wide")

# ----- Settings -----
LIVE_CSV = "neonatal_live.csv"
HISTORICAL_EXCEL = "neonatal_incubator_with_actions.xlsx"
REFRESH_SECONDS = 120  # auto-refresh every 2 minutes
PREDICT_MINUTES = 10  # predict next N minutes

# Thresholds
TEMP_LOW, TEMP_HIGH = 36.5, 37.2
HUM_LOW, HUM_HIGH = 50, 65
HR_LOW, HR_HIGH = 120, 160

# ----- Page Header -----
st.title("🍼 Neonatal Baby Incubator Dashboard")
st.markdown(f"Auto-refresh every **{REFRESH_SECONDS} seconds**.")

# ----- Mode selection -----
mode = st.sidebar.radio("Mode:", ["Historical Analysis", "Live / Simulation"])

# ----- Helper functions -----
def check_alerts(row):
    alerts = []
    if row['temperature'] < TEMP_LOW or row['temperature'] > TEMP_HIGH:
        alerts.append("Temperature")
    if row['humidity'] < HUM_LOW or row['humidity'] > HUM_HIGH:
        alerts.append("Humidity")
    if row['heart_rate'] < HR_LOW or row['heart_rate'] > HR_HIGH:
        alerts.append("Heart Rate")
    return alerts

def load_historical():
    if not os.path.exists(HISTORICAL_EXCEL):
        st.error(f"Historical data not found: {HISTORICAL_EXCEL}")
        return None
    df = pd.read_excel(HISTORICAL_EXCEL, sheet_name='readings', engine='openpyxl')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['alerts'] = df.apply(check_alerts, axis=1)
    return df

def load_live():
    if os.path.exists(LIVE_CSV):
        df = pd.read_csv(LIVE_CSV)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    else:
        st.info("Live CSV not found, using last historical data fallback.")
        df = load_historical()
    if df is None:
        st.stop()
    df['alerts'] = df.apply(check_alerts, axis=1)
    return df

# ----- Main Logic -----
if mode == "Historical Analysis":
    st.subheader("📊 Historical Data")
    df = load_historical()
elif mode == "Live / Simulation":
    st.subheader("⏱ Live Data / Simulation")
    df = load_live()

# ----- Latest readings & device actions -----
latest = df.iloc[-1]
st.subheader("Latest Reading")
st.metric("Temperature (°C)", f"{latest.temperature:.2f}")
st.metric("Humidity (%)", f"{latest.humidity:.1f}")
st.metric("Weight (kg)", f"{latest.weight:.3f}")
st.metric("Heart Rate (bpm)", f"{int(latest.heart_rate)}")

st.subheader("Device Actions (Latest)")
cols = st.columns(3)
cols[0].markdown(f"**Fan**: {'🔴 ON' if int(latest.get('fan_status',0)) else '🟢 OFF'}")
cols[1].markdown(f"**Heater**: {'🔴 ON' if int(latest.get('heater_status',0)) else '🟢 OFF'}")
cols[2].markdown(f"**Alarm**: {'🔴 ACTIVE' if int(latest.get('alarm_status',0)) else '🟢 OK'}")

# ----- Individual Graphs -----
st.subheader("Parameter Graphs (Zoomable)")
params = ['temperature','humidity','heart_rate','weight']
for param in params:
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(df['timestamp'], df[param], marker='o', color='blue', label=param.capitalize())
    # highlight alerts
    for i, row in df.iterrows():
        if param.lower() in [a.lower() for a in row['alerts']]:
            ax.plot(row['timestamp'], row[param], marker='o', color='red', markersize=8)
    ax.set_xlabel("Time")
    ax.set_ylabel(param.capitalize())
    ax.grid(True)
    st.pyplot(fig)

# ----- Alerts Table -----
st.subheader("Recent Alerts")
alerts_df = df[df['alerts'].map(len) > 0]
if not alerts_df.empty:
    alerts_df['alerts_text'] = alerts_df['alerts'].apply(lambda x: ", ".join(x))
    st.dataframe(alerts_df[['timestamp','temperature','humidity','heart_rate','alerts_text']].tail(20))
    # Show emergency warning for latest alerts
    latest_alerts = alerts_df.iloc[-1]['alerts'] if not alerts_df.empty else []
    if latest_alerts:
        st.warning(f"⚠ Emergency Alert: {', '.join(latest_alerts)} at {alerts_df.iloc[-1]['timestamp']}")
else:
    st.success("✅ All parameters normal")

# ----- Simple Forecast -----
st.subheader("Forecast (Next Minutes)")
if 'time_idx' not in df.columns:
    df['time_idx'] = np.arange(len(df))
X = df['time_idx'].values
if len(X) >= 5:
    # Weight
    wcoef = polyfit(X, df['weight'].values, 1)
    future_idx = np.arange(X[-1]+1, X[-1]+1+PREDICT_MINUTES)
    pred_weight = polyval(wcoef, future_idx)
    # Heart Rate
    hrcoef = polyfit(X, df['heart_rate'].values, 1)
    pred_hr = polyval(hrcoef, future_idx)
    future_timestamps = [df['timestamp'].iloc[-1] + timedelta(minutes=i+1) for i in range(PREDICT_MINUTES)]
    pred_df = pd.DataFrame({
        'timestamp': future_timestamps,
        'predicted_weight': np.round(pred_weight,3),
        'predicted_heart_rate': np.round(pred_hr,1)
    })
    st.dataframe(pred_df)
else:
    st.info("Not enough points to predict (need >=5 readings).")

# ----- Overall Progress Score -----
alerts_count = alerts_df.shape[0] if not alerts_df.empty else 0
progress_score = max(0, 100 - alerts_count*0.5)
st.subheader("Overall Progress Score")
st.metric("Progress Score (0-100)", int(progress_score))
st.markdown("**Note:** This is a demo metric. Always rely on medical staff for decisions.")

# ----- Auto-refresh -----
st.experimental_rerun()  # refresh every REFRESH_SECONDS
"""

"""
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
SIMULATION_INTERVAL = 120  # seconds (2 minutes refresh)

# ---------------- THRESHOLDS ----------------
TEMP_LOW, TEMP_HIGH = 36.5, 37.2
HUM_LOW, HUM_HIGH = 50, 65
HR_LOW, HR_HIGH = 120, 160

st.set_page_config(page_title="Neonatal Incubator Dashboard", layout="wide")
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

    st.info("Simulating live data. App refreshes every 2 minutes automatically.")

    while True:  # continuously refresh every 2 min
        if df_live.empty:
            last_weight = 3.0
        else:
            last_weight = df_live.iloc[-1]['weight']

        # Generate simulated live data
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
