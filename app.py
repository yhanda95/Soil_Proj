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

# ================= LOAD MODEL (WEIGHTS METHOD) =================
@st.cache_resource
def load_model():
    WEIGHTS_PATH = "model.weights.h5"

    # 🔽 Download from Drive if not exists
    if not os.path.exists(WEIGHTS_PATH):
        st.info("Downloading model weights...")
        url = "https://drive.google.com/file/d/1DOkjGe0GowFu4NBrmyhC2iZWHraysO7I/view?usp=sharing"
        gdown.download(url, WEIGHTS_PATH, quiet=False)

    # 🔥 Rebuild SAME model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),

        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(4, activation='softmax')  # 🔴 CHANGE if needed
    ])

    model.load_weights(WEIGHTS_PATH)

    return model

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
        st.warning("No data found")

# =====================================================
# 📈 ANALYTICS
# =====================================================
elif page == "📈 Analytics":

    st.subheader("Sensor Data Analytics")

    data = ref.get()

    if data:
        df = pd.DataFrame(list(data.values())).tail(50)

        fig = px.line(df, y=["soil", "temp", "hum"])
        st.plotly_chart(fig, use_container_width=True)

        if df["hum"].mean() > 70:
            st.warning("High humidity trend → fungal risk")

        if df["soil"].min() < 1500:
            st.warning("Low soil moisture detected")

    else:
        st.warning("No data available")

# =====================================================
# 🤖 CHATBOT
# =====================================================
elif page == "🤖 AI Chatbot":

    st.subheader("AI Farming Assistant")

    user_input = st.text_area("Ask your question")

    data = ref.get()

    if data:
        latest = list(data.values())[-1]
        soil = latest.get('soil', "unknown")
        temp = latest.get('temp', "unknown")
        hum = latest.get('hum', "unknown")
    else:
        soil, temp, hum = "unknown", "unknown", "unknown"

    if user_input:
        prompt = f"""
        You are an agriculture expert.

        Soil: {soil}
        Temp: {temp}
        Humidity: {hum}

        Question: {user_input}

        Give short practical advice.
        """

        response = chat_model.generate_content(prompt)
        st.success(response.text)
