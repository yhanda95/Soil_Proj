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
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(dict(st.secrets["firebase"]))

            firebase_admin.initialize_app(cred, {
                "databaseURL": "https://soilproj-eac88-default-rtdb.europe-west1.firebasedatabase.app/"
            })
        return True
    except Exception as e:
        st.error("❌ Firebase init failed")
        st.write(e)
        return False

firebase_ok = init_firebase()

if not firebase_ok:
    st.stop()

ref = db.reference("sensor")

# ================= GEMINI =================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
chat_model = genai.GenerativeModel("gemini-1.5-flash")

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

    try:
        data = ref.get()
    except Exception as e:
        st.error("❌ Firebase fetch error")
        st.write(e)
        st.stop()

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

        if soil < 1500:
            st.error("Soil is DRY → Irrigation needed")
        elif soil < 3000:
            st.warning("Soil moisture is MODERATE")
        else:
            st.success("Soil moisture is GOOD")

        if hum > 80 and 20 < temp < 30:
            st.error("High fungal disease risk ⚠")
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

    try:
        data = ref.get()
    except Exception as e:
        st.error("❌ Firebase fetch error")
        st.write(e)
        st.stop()

    if data:
        df = pd.DataFrame(list(data.values())).tail(50)

        st.markdown("### 📊 Trends")
        fig = px.line(df, y=["soil", "temp", "hum"])
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📌 Insights")

        if df["hum"].mean() > 70:
            st.warning("High humidity trend → fungal risk")

        if df["soil"].min() < 1500:
            st.warning("Low soil moisture detected")

    else:
        st.warning("No data available")

# =====================================================
# 🤖 AI CHATBOT
# =====================================================
elif page == "🤖 AI Chatbot":

    st.subheader("AI Farming Assistant")

    user_input = st.text_area("Ask your farming question")

    try:
        data = ref.get()
    except:
        data = None

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

Give short, practical farming advice.
"""

            try:
                response = chat_model.generate_content(prompt)
                st.success(response.text)
            except Exception as e:
                st.error("❌ Chatbot error")
                st.write(e)
