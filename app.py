import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import pandas as pd
import plotly.express as px
import firebase_admin
from firebase_admin import credentials, db
from streamlit_autorefresh import st_autorefresh
import google.generativeai as genai
import gdown
import os

# ================= CONFIG =================
st.set_page_config(page_title="Smart Agriculture", layout="wide")
st_autorefresh(interval=3000, key="refresh")

# ================= LOAD MODEL FROM DRIVE =================
@st.cache_resource
def load_model():
    MODEL_PATH = "final_model.keras"

    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model from Drive...")
        url = "https://drive.google.com/file/d/1DGsTskifcu2h1KdvSJjRJRPXKd_wS3E3/view?usp=sharing"
        gdown.download(url, MODEL_PATH, quiet=False)

    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ================= LOAD LABELS =================
with open("class_indices.json") as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}

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

# ================= IMAGE PREPROCESS =================
def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ================= UI =================
st.title("🌱 Smart Agriculture System")

page = st.sidebar.radio("Navigation", [
    "🌿 Disease Detection",
    "📊 Live Data",
    "📈 Analytics",
    "🤖 AI Chatbot"
])

# =====================================================
# 🌿 DISEASE DETECTION
# =====================================================
if page == "🌿 Disease Detection":

    st.subheader("Upload Leaf Image")

    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, use_column_width=True)

        processed = preprocess(img)
        pred = model.predict(processed)

        idx = np.argmax(pred)
        confidence = np.max(pred) * 100

        st.success(f"Prediction: {idx_to_class[idx]}")
        st.info(f"Confidence: {confidence:.2f}%")

# =====================================================
# 📊 LIVE DATA
# =====================================================
elif page == "📊 Live Data":

    st.subheader("Real-Time Sensor Data")

    data = ref.get()

    if data:
        latest = list(data.values())[-1]

        soil = latest.get('soil', 0)
        temp = latest.get('temp', 0)
        hum = latest.get('hum', 0)

        c1, c2, c3 = st.columns(3)

        c1.metric("🌱 Soil", soil)
        c2.metric("🌡 Temperature", temp)
        c3.metric("💧 Humidity", hum)

        st.markdown("### ⚠ Condition Analysis")

        if soil < 1500:
            st.error("Soil is DRY → Irrigation needed")
        elif soil < 3000:
            st.warning("Soil moisture is MODERATE")
        else:
            st.success("Soil moisture is GOOD")

        if hum > 80 and 20 < temp < 30:
            st.error("High risk of fungal disease")
        else:
            st.success("No immediate disease risk")

    else:
        st.warning("No data found in Firebase")

# =====================================================
# 📈 ANALYTICS
# =====================================================
elif page == "📈 Analytics":

    st.subheader("Sensor Data Analytics")

    data = ref.get()

    if data:
        df = pd.DataFrame(list(data.values()))
        df = df.tail(50)

        st.markdown("### 📊 Trends")
        fig1 = px.line(df, y=["soil", "temp", "hum"])
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("### 🌱 Soil Moisture")
        fig2 = px.area(df, y="soil")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### 💧 Humidity Distribution")
        fig3 = px.histogram(df, x="hum")
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("### 📌 Insights")

        if df["hum"].mean() > 70:
            st.warning("High humidity trend → fungal diseases possible")

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
