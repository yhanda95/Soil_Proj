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
st_autorefresh(interval=5000, key="refresh")

# ================= MODEL LOADING =================
@st.cache_resource
def load_model():

    WEIGHTS_PATH = "model.weights.h5"

    # ✅ Download ONLY if not present
    if not os.path.exists(WEIGHTS_PATH):

        st.info("Downloading model weights...")

        url = "https://drive.google.com/file/d/1DOkjGe0GowFu4NBrmyhC2iZWHraysO7I/view?usp=share_link"

        gdown.download(url, WEIGHTS_PATH, quiet=False)

    # ✅ Check file
    if not os.path.exists(WEIGHTS_PATH):
        st.error("❌ Model download failed")
        st.stop()

    size = os.path.getsize(WEIGHTS_PATH)

    if size < 5_000_000:  # <5MB means broken
        st.error("❌ Corrupted model file. Fix Google Drive link.")
        st.stop()

    # 🔥 Rebuild SAME architecture (VERY IMPORTANT)
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

        tf.keras.layers.Dense(4, activation='softmax')  # 🔴 CHANGE IF NEEDED
    ])

    # ✅ Load weights safely
    try:
        model.load_weights(WEIGHTS_PATH)
    except Exception as e:
        st.error("❌ Model loading failed")
        st.write(e)
        st.stop()

    return model


model = load_model()

# ================= LABELS =================
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

# ================= PREPROCESS =================
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

    file = st.file_uploader("Upload leaf image", type=["jpg", "png", "jpeg"])

    if file:
        img = Image.open(file).convert("RGB")
        st.image(img)

        pred = model.predict(preprocess(img))

        idx = np.argmax(pred)
        confidence = np.max(pred) * 100

        st.success(f"Prediction: {idx_to_class[idx]}")
        st.info(f"Confidence: {confidence:.2f}%")

# =====================================================
# 📊 LIVE DATA
# =====================================================
elif page == "📊 Live Data":

    data = ref.get()

    if data:
        latest = list(data.values())[-1]

        soil = latest.get('soil', 0)
        temp = latest.get('temp', 0)
        hum = latest.get('hum', 0)

        c1, c2, c3 = st.columns(3)
        c1.metric("🌱 Soil", soil)
        c2.metric("🌡 Temp", temp)
        c3.metric("💧 Humidity", hum)

        if soil < 1500:
            st.error("Soil too dry")
        elif soil < 3000:
            st.warning("Moderate moisture")
        else:
            st.success("Good moisture")

        if hum > 80 and 20 < temp < 30:
            st.error("⚠ High fungal risk")

# =====================================================
# 📈 ANALYTICS
# =====================================================
elif page == "📈 Analytics":

    data = ref.get()

    if data:
        df = pd.DataFrame(list(data.values())).tail(50)

        st.plotly_chart(px.line(df, y=["soil", "temp", "hum"]))

        if df["hum"].mean() > 70:
            st.warning("High humidity trend")

# =====================================================
# 🤖 CHATBOT
# =====================================================
elif page == "🤖 AI Chatbot":

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
        """

        response = chat_model.generate_content(prompt)
        st.success(response.text)
