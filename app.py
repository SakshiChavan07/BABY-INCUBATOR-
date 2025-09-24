# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from numpy import polyfit, polyval
import os

st.set_page_config(page_title="🍼 Neonatal Incubator Live Dashboard", layout="wide")

# ----- Settings -----
CSV_FILE = "neonatal_live.csv"
EXCEL_FILE = "neonatal_incubator_with_actions.xlsx"
REFRESH_SECONDS = 60  # auto refresh period (1 minute)
PREDICT_MINUTES = 10  # predict next N minutes

TEMP_LOW, TEMP_HIGH = 36.5, 37.2
HUM_LOW, HUM_HIGH = 50, 65
HR_LOW, HR_HIGH = 120, 160

# ----- Auto refresh (simple) -----
st.experimental_set_query_params()  # no-op that helps with rerun behavior
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = 0

# Provide a manual refresh button and info about auto-refresh
st.title("🍼 Neonatal Incubator — Live Dashboard")
st.markdown(f"Auto-refresh every **{REFRESH_SECONDS} seconds**. (Make sure `serial_to_csv.py` is running if you want real live data.)")

col1, col2 = st.columns([1,3])
with col1:
    if st.button("Refresh now"):
        st.experimental_rerun()
with col2:
    st.write("Live source:", "CSV (serial feed)" if os.path.exists(CSV_FILE) else "Excel fallback")

# ----- Load data -----
def load_data():
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
    else:
        # fallback to excel readings sheet
        df = pd.read_excel(EXCEL_FILE, sheet_name='readings', engine='openpyxl')
    # normalize columns
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    else:
        # if CSV missing timestamp column (unlikely), create one (current time)
        df['timestamp'] = pd.to_datetime(df.index, unit='m', origin='unix')
    # compute actions/alerts if missing
    for col in ['fan_status','heater_status','alarm_status']:
        if col not in df.columns:
            df[col] = 0
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

df = load_data()

# If no data
if df.empty:
    st.error("No data available. Run serial_to_csv.py (connected to your Arduino/ESP32), or upload the Excel.")
    st.stop()

# Show latest row
latest = df.iloc[-1]
st.subheader("Latest Reading")
st.metric("Temperature (°C)", f"{latest.temperature:.2f}")
st.metric("Humidity (%)", f"{latest.humidity:.1f}")
st.metric("Weight (kg)", f"{latest.weight:.3f}")
st.metric("Heart Rate (bpm)", f"{int(latest.heart_rate)}")

# Show device actions
st.subheader("Device Actions (Latest)")
cols = st.columns(3)
cols[0].markdown(f"**Fan**: {'🔴 ON' if int(latest.get('fan_status',0))==1 else '🟢 OFF'}")
cols[1].markdown(f"**Heater**: {'🔴 ON' if int(latest.get('heater_status',0))==1 else '🟢 OFF'}")
cols[2].markdown(f"**Alarm**: {'🔴 ACTIVE' if int(latest.get('alarm_status',0))==1 else '🟢 OK'}")

# Parameter graphs (last N points for clarity)
st.subheader("Parameter Graphs")
to_plot = df.set_index('timestamp')[
    ['temperature','humidity','heart_rate','weight']
].tail(300)  # limit to last 300 minutes for speed
st.line_chart(to_plot['temperature'].rename("Temperature (°C)").to_frame())
st.line_chart(to_plot['humidity'].rename("Humidity (%)").to_frame())
st.line_chart(to_plot['heart_rate'].rename("Heart Rate (bpm)").to_frame())
st.line_chart(to_plot['weight'].rename("Weight (kg)").to_frame())

# Alerts table (last 20)
st.subheader("Recent Alerts")
alerts = df[(df['temperature'] < TEMP_LOW) | (df['temperature'] > TEMP_HIGH) |
            (df['humidity'] < HUM_LOW) | (df['humidity'] > HUM_HIGH) |
            (df['heart_rate'] < HR_LOW) | (df['heart_rate'] > HR_HIGH)]
st.dataframe(alerts[['timestamp','temperature','humidity','heart_rate']].tail(20))

# Simple linear prediction (next PREDICT_MINUTES)
st.subheader("Simple Forecast (next minutes)")
# create time_idx if missing
if 'time_idx' not in df.columns:
    df['time_idx'] = np.arange(len(df))
X = df['time_idx'].values
if len(X) >= 5:
    # weight
    wcoef = polyfit(X, df['weight'].values, 1)
    future_idx = np.arange(X[-1]+1, X[-1]+1+PREDICT_MINUTES)
    pred_weight = polyval(wcoef, future_idx)
    # hr
    hrcoef = polyfit(X, df['heart_rate'].values, 1)
    pred_hr = polyval(hrcoef, future_idx)
    future_timestamps = [df['timestamp'].iloc[-1] + timedelta(minutes=i+1) for i in range(PREDICT_MINUTES)]
    pred_df = pd.DataFrame({
        'timestamp': future_timestamps,
        'predicted_weight': np.round(pred_weight,3),
        'predicted_heart_rate': np.round(pred_hr,1)
    })
    st.write("Predictions (next timestamps):")
    st.dataframe(pred_df)
    # show simple chart combining last hour + predictions
    recent = df.set_index('timestamp')[['weight','heart_rate']].tail(60)
    combined_weight = pd.concat([recent['weight'], pd.Series(pred_df['predicted_weight'].values, index=pred_df['timestamp'])])
    st.line_chart(combined_weight.rename("Weight (kg)").to_frame())
    combined_hr = pd.concat([recent['heart_rate'], pd.Series(pred_df['predicted_heart_rate'].values, index=pred_df['timestamp'])])
    st.line_chart(combined_hr.rename("Heart Rate (bpm)").to_frame())
else:
    st.info("Not enough points to predict (need >=5 readings).")

# Progress score (simple)
alerts_count = alerts.shape[0]
progress_score = max(0, 100 - alerts_count * 0.5)
st.subheader("Overall Progress Score")
st.metric("Progress Score (0-100)", int(progress_score))
st.markdown("**Note:** This is a simple demo metric. Always rely on medical staff for decisions.")

# Auto-refresh note & rerun
st.write(f"App updates every {REFRESH_SECONDS} seconds. Keep this page open to see new readings.")
st.experimental_rerun()  # Re-run the script so it will refresh (browser refresh will also work)
