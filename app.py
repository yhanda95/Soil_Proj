import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

import firebase_admin
from firebase_admin import credentials, db

from transformers import pipeline

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Smart Agriculture AI 🌿🤖📡", layout="centered")

# ---------------- FIREBASE INIT ----------------
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_key.json")  # 🔥 your firebase key file

    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://YOUR_PROJECT.firebaseio.com/"  # 🔥 replace this
    })

def get_sensor_data():
    ref = db.reference("sensor")
    return ref.get()

# ---------------- CHATBOT ----------------
@st.cache_resource
def load_chatbot():
    return pipeline("text-generation", model="distilgpt2")

chatbot = load_chatbot()

def get_response(text):
    result = chatbot(text, max_length=120, num_return_sequences=1)
    return result[0]["generated_text"]

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras")

model = load_model()

# ---------------- DISEASE INFO ----------------
with open("disease_info.json", "r") as f:
    disease_info = json.load(f)

class_names = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_healthy"
]

# ---------------- PREPROCESS ----------------
def preprocess(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image

# ---------------- SIDEBAR ----------------
page = st.sidebar.selectbox("Select Page", [
    "🌿 Disease Detection",
    "📡 Live Sensor Data",
    "🤖 AI Chatbot"
])

# =====================================================
# 🌿 PAGE 1 - DISEASE DETECTION
# =====================================================
if page == "🌿 Disease Detection":

    st.title("🌿 Plant Disease Detection")

    uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image")

        processed = preprocess(image)
        prediction = model.predict(processed)

        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        st.subheader(f"🧠 Prediction: {predicted_class}")
        st.write(f"Confidence: {confidence*100:.2f}%")

        if predicted_class in disease_info:
            st.subheader("🦠 Cause")
            st.write(disease_info[predicted_class]["cause"])

            st.subheader("🛡 Prevention")
            st.write(disease_info[predicted_class]["prevention"])

# =====================================================
# 📡 PAGE 2 - FIREBASE SENSOR DATA
# =====================================================
elif page == "📡 Live Sensor Data":

    st.title("📡 Live Farm Sensor Data")

    data = get_sensor_data()

    if data:
        st.metric("🌡 Temperature", f"{data['temperature']} °C")
        st.metric("💧 Humidity", f"{data['humidity']} %")
        st.metric("🌱 Soil Moisture", f"{data['soil_moisture']} %")
    else:
        st.warning("No sensor data available. Check Firebase connection.")

# =====================================================
# 🤖 PAGE 3 - CHATBOT
# =====================================================
elif page == "🤖 AI Chatbot":

    st.title("🤖 Farming AI Chatbot")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask anything...")

    if user_input:

        st.chat_message("user").write(user_input)
        st.session_state.chat.append({"role": "user", "content": user_input})

        with st.spinner("Thinking..."):
            reply = get_response(user_input)

        st.chat_message("assistant").write(reply)
        st.session_state.chat.append({"role": "assistant", "content": reply})
