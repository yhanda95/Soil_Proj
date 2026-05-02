import streamlit as st
import pandas as pd
import plotly.express as px
import firebase_admin
from firebase_admin import credentials, db
from streamlit_autorefresh import st_autorefresh
import google.generativeai as genai

# ================= CONFIG =================
st.set_page_config(page_title="Smart Agriculture", layout="wide")
st_autorefresh(interval=5000, key="refresh")

# ================= FIREBASE =================
@st.cache_resource
def init_firebase():
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://soilproj-eac88-default-rtdb.europe-west1.firebasedatabase.app/'
    })

if not firebase_admin._apps:
    init_firebase()

ref = db.reference("sensor")

# ================= GEMINI =================
genai.configure(api_key="AIzaSyDJvyVrdsD_DxzCyzFbf6rm-h5br7ksMlc")
chat_model = genai.GenerativeModel("gemini-pro")

# ================= UI =================
st.title("🌱 Smart Agriculture System")

page = st.sidebar.radio("Navigation", [
    "📊 Live Data",
    "📈 Analytics",
    "🤖 AI Chatbot"
])

# =====================================================
# 📊 LIVE DATA
# =====================================================
if page == "📊 Live Data":

    st.subheader("Real-Time Sensor Data")

    data = ref.get()

    if data:
        latest = list(data.values())[-1]

        soil = latest.get('soil', 0)
        temp = latest.get('temp', 0)
        hum = latest.get('hum', 0)

        c1, c2, c3 = st.columns(3)

        c1.metric("🌱 Soil Moisture", soil)
        c2.metric("🌡 Temperature", temp)
        c3.metric("💧 Humidity", hum)

        st.markdown("### ⚠ Smart Prediction")

        # 🌱 Soil logic
        if soil < 1500:
            st.error("Soil is DRY → Irrigation needed immediately")
        elif soil < 3000:
            st.warning("Soil moisture is MODERATE")
        else:
            st.success("Soil moisture is GOOD")

        # 🦠 Disease risk logic
        if hum > 80 and 20 < temp < 30:
            st.error("High risk of fungal disease ⚠")
        elif hum > 70:
            st.warning("Moderate disease risk")
        else:
            st.success("Low disease risk")

    else:
        st.warning("No data found in Firebase")

# =====================================================
# 📈 ANALYTICS
# =====================================================
elif page == "📈 Analytics":

    st.subheader("Sensor Data Analytics")

    data = ref.get()

    if data:
        df = pd.DataFrame(list(data.values())).tail(50)

        st.markdown("### 📊 Trends")
        st.plotly_chart(px.line(df, y=["soil", "temp", "hum"]),
                        use_container_width=True)

        st.markdown("### 📌 Insights")

        if df["hum"].mean() > 70:
            st.warning("High humidity trend → fungal disease risk")

        if df["soil"].min() < 1500:
            st.warning("Low soil moisture detected → irrigation needed")

    else:
        st.warning("No data available")

# =====================================================
# 🤖 AI CHATBOT
# =====================================================
elif page == "🤖 AI Chatbot":

    st.subheader("AI Farming Assistant")

    user_input = st.text_area("Ask your farming question")

    data = ref.get()

    if data:
        latest = list(data.values())[-1]
        soil = latest.get('soil', "unknown")
        temp = latest.get('temp', "unknown")
        hum = latest.get('hum', "unknown")
    else:
        soil, temp, hum = "unknown", "unknown", "unknown"

    if user_input:
        with st.spinner("Thinking..."):

            prompt = f"""
            You are an expert agricultural advisor.

            Current conditions:
            Soil Moisture: {soil}
            Temperature: {temp}
            Humidity: {hum}

            Question: {user_input}

            Give clear, short, practical farming advice.
            """

            response = chat_model.generate_content(prompt)
            st.success(response.text)
