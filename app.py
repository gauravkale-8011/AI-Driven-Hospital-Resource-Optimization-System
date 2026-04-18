import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Hospital AI System", layout="wide")

# ---------------- LOAD MODELS ----------------
model_patients = pickle.load(open("model/model_patients.pkl", "rb"))
model_emergency = pickle.load(open("model/model_emergency.pkl", "rb"))
model_icu = pickle.load(open("model/model_icu.pkl", "rb"))


# ---------------- WEATHER FUNCTION ----------------
def get_temperature(lat, lon, selected_date):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max&timezone=auto"

    try:
        response = requests.get(url)
        data = response.json()

        dates = data['daily']['time']
        temps = data['daily']['temperature_2m_max']

        for i in range(len(dates)):
            if dates[i] == str(selected_date):
                return temps[i]

        return temps[0]  # fallback

    except:
        return 30  # safe fallback


# ---------------- CSS ----------------
st.markdown("""
<style>
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    margin: 10px;
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">🏥 Hospital AI Decision System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict • Optimize • Decide</div>', unsafe_allow_html=True)

# ---------------- INPUT ----------------
col1, col2 = st.columns(2)

today = datetime.today().date()
max_date = today + timedelta(days=7)

with col1:
    date = st.date_input("📅 Select Date", min_value=today, max_value=max_date)

with col2:
    city = st.selectbox("📍 Select City", ["Mumbai", "Delhi", "Pune"])

# ---------------- CITY COORDS ----------------
city_coords = {
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.7041, 77.1025),
    "Pune": (18.5204, 73.8567)
}

lat, lon = city_coords[city]

# ---------------- TEMPERATURE ----------------
temperature = get_temperature(lat, lon, date)
st.info(f"🌡️ Temperature in {city}: {temperature}°C")

# ---------------- WHAT-IF SIMULATOR ----------------
st.sidebar.title("🎛️ What-if Simulator")

staff_available = st.sidebar.slider("👨‍⚕️ Staff Available", 10, 50, 25)
beds_available = st.sidebar.slider("🛏️ Beds Available", 10, 100, 50)

# ---------------- PREDICT ----------------
if st.button("🚀 Run Simulation"):

    # Extract features
    day = date.day
    month = date.month
    day_of_week = date.weekday()

    is_weekend = 1 if day_of_week >= 5 else 0
    is_holiday = 0

    # Initial lag values (can improve later)
    lag_1 = 50
    lag_2 = 48
    rolling_mean = 49

    # Dummy inputs
    emergency_cases = 10
    icu_needed = 5

    input_data = pd.DataFrame([[
        emergency_cases,
        icu_needed,
        staff_available,
        temperature,
        is_holiday,
        is_weekend,
        day,
        month,
        day_of_week,
        lag_1,
        lag_2,
        rolling_mean
    ]], columns=[
        'emergency_cases',
        'icu_needed',
        'staff_available',
        'temperature',
        'is_holiday',
        'is_weekend',
        'day',
        'month',
        'day_of_week',
        'lag_1',
        'lag_2',
        'rolling_mean'
    ])

    # ---------------- ML + FALLBACK ----------------
    try:
        patients = int(model_patients.predict(input_data)[0])
        emergency = int(model_emergency.predict(input_data)[0])
        icu = int(model_icu.predict(input_data)[0])

    except Exception as e:
        st.warning("⚠️ Using fallback system")

        patients = int(model_patients.predict(input_data)[0])
        emergency = int(patients * 0.3)
        icu = int(emergency * 0.3)

    # Staff calculation
    staff_required = int(patients / 2)

    # ---------------- METRICS ----------------
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("👥 Patients", patients)
    col2.metric("🚑 Emergency", emergency)
    col3.metric("🏥 ICU", icu)
    col4.metric("👨‍⚕️ Staff Needed", staff_required)

    # ---------------- RISK SCORE ----------------
    risk_score = int((patients / beds_available) * 100)

    st.subheader("⚠️ Risk Score")
    st.progress(min(risk_score, 100))

    if risk_score < 40:
        st.success(f"🟢 Low Risk ({risk_score})")
    elif risk_score < 70:
        st.warning(f"🟡 Moderate Risk ({risk_score})")
    else:
        st.error(f"🔴 High Risk ({risk_score})")

    # ---------------- OPTIMIZATION ----------------
    st.subheader("⚙️ Optimization Suggestions")

    if staff_required > staff_available:
        st.error(f"❗ Need {staff_required - staff_available} more staff")
    else:
        st.success("✅ Staff sufficient")

    if patients > beds_available:
        st.warning(f"⚠️ Shortage of {patients - beds_available} beds")
    else:
        st.success("✅ Beds sufficient")

    # ---------------- WHAT-IF ----------------
    st.subheader("🎛️ What-if Analysis")

    if staff_available < staff_required:
        st.error("Overload likely → Increase staff")
    else:
        st.success("System stable")

    # ---------------- FORECAST ----------------
    st.subheader("📈 7-Day Forecast")

    future_days = []
    predictions = []

    temp_lag_1 = lag_1
    temp_lag_2 = lag_2

    for i in range(7):
        future_day = (day + i) % 31 or 1

        input_data.loc[0, 'day'] = future_day
        input_data.loc[0, 'lag_1'] = int(temp_lag_1)
        input_data.loc[0, 'lag_2'] = int(temp_lag_2)

        pred = model_patients.predict(input_data)[0]

        # update lag dynamically
        temp_lag_2 = temp_lag_1
        temp_lag_1 = pred

        future_days.append(f"Day {future_day}")
        predictions.append(pred)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=future_days,
        y=predictions,
        mode='lines+markers',
        line=dict(color='cyan')
    ))

    fig.update_layout(template="plotly_dark")

    st.plotly_chart(fig, use_container_width=True)