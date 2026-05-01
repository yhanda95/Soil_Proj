import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import firebase_admin
from firebase_admin import credentials, db
import json
import google.generativeai as genai
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

# ================= CONFIG =================
st.set_page_config(page_title="Smart Agriculture", layout="wide")

# 🔁 AUTO REFRESH
st_autorefresh(interval=2000, key="refresh")

# ================= LOAD MODEL (FIXED) =================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fixed_model.h5", compile=False)

model = load_model()

# ================= LOAD JSON =================
with open("class_indices.json") as f:
    class_indices = json.load(f)

with open("disease_info.json") as f:
    disease_info = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}

# ================= FIREBASE =================
@st.cache_resource
def init_firebase():
    cred = credentials.Certificate("firebase_key.json")  # or secrets if deployed
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'YOUR_FIREBASE_URL'
    })

if not firebase_admin._apps:
    init_firebase()

ref = db.reference('sensor')

# ================= GEMINI =================
genai.configure(api_key="YOUR_GEMINI_API_KEY")
gemini = genai.GenerativeModel("gemini-pro")

# ================= PREPROCESS =================
def preprocess(img):
    img = img.resize((224, 224))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ================= HEADER =================
st.markdown("""
<h1 style='text-align:center; color:#2E8B57;'>
🌱 Smart Agriculture Dashboard
</h1>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
page = st.sidebar.radio("Navigation", [
    "🌿 Disease Detection",
    "📊 Live Data",
    "📈 Analytics",
    "🤖 AI Chatbot"
])

# =====================================================
# 🌿 PAGE 1: DISEASE DETECTION
# =====================================================
if page == "🌿 Disease Detection":

    st.subheader("Upload Leaf Image")

    file = st.file_uploader("Choose image")

    if file:
        img = Image.open(file)
        st.image(img, use_column_width=True)

        processed = preprocess(img)
        pred = model.predict(processed)

        idx = np.argmax(pred)
        disease = idx_to_class[idx]
        conf = np.max(pred) * 100

        st.success(f"{disease} ({conf:.2f}%)")

        info = disease_info[disease]

        st.markdown("### 🦠 Cause")
        st.write(info["cause"])

        st.markdown("### 🛡 Prevention")
        st.write(info["prevention"])

# =====================================================
# 📊 PAGE 2: LIVE DATA
# =====================================================
elif page == "📊 Live Data":

    st.subheader("Real-Time Sensor Data")

    data = ref.get()

    if data:
        latest_key = list(data.keys())[-1]
        latest = data[latest_key]

        soil = latest['soil']
        temp = latest['temp']
        hum = latest['hum']

        c1, c2, c3 = st.columns(3)

        c1.metric("🌱 Soil", soil)
        c2.metric("🌡 Temp", temp)
        c3.metric("💧 Humidity", hum)

        st.markdown("---")

        # Soil status
        if soil < 1500:
            st.error("🔴 Soil is DRY")
        elif soil < 3000:
            st.warning("🟡 Soil is MODERATE")
        else:
            st.success("🟢 Soil is WET")

        st.markdown("### ⚠ Prediction")

        if hum > 80 and 20 < temp < 30:
            st.error("High fungal disease risk")
        elif soil < 1500:
            st.warning("Irrigation required")
        else:
            st.success("Conditions optimal")

# =====================================================
# 📈 PAGE 3: ANALYTICS
# =====================================================
elif page == "📈 Analytics":

    st.subheader("Sensor Analytics")

    data = ref.get()

    if data:
        rows = [v for v in data.values()]
        df = pd.DataFrame(rows)

        df = df.tail(50)

        fig1 = px.line(df, y=["soil","temp","hum"], title="Sensor Trends")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.area(df, y="soil", title="Soil Moisture Trend")
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.histogram(df, x="hum", title="Humidity Distribution")
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("### 📊 Insights")

        if df["hum"].mean() > 70:
            st.warning("High humidity trend detected")

        if df["soil"].min() < 1500:
            st.warning("Soil dryness detected")

# =====================================================
# 🤖 PAGE 4: CHATBOT
# =====================================================
elif page == "🤖 AI Chatbot":

    st.subheader("AI Farming Assistant")

    user_input = st.text_area("Ask your question")

    data = ref.get()

    if data:
        latest_key = list(data.keys())[-1]
        latest = data[latest_key]

        soil = latest['soil']
        temp = latest['temp']
        hum = latest['hum']

    if user_input:
        with st.spinner("Thinking..."):
            prompt = f"""
            Soil: {soil}
            Temperature: {temp}
            Humidity: {hum}

            Question: {user_input}

            Give actionable farming advice.
            """

            response = gemini.generate_content(prompt)
            st.success(response.text)
