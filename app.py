import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

from transformers import pipeline

import firebase_admin
from firebase_admin import credentials, db

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Plant Disease Detection 🌿", layout="centered")

# ---------------- FIREBASE INIT ----------------
if not firebase_admin._apps:

    firebase_config = dict(st.secrets["firebase"])
    firebase_config["private_key"] = firebase_config["private_key"].replace("\\n", "\n")

    cred = credentials.Certificate(firebase_config)

    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://soilproj-eac88-default-rtdb.europe-west1.firebasedatabase.app/"
    })

# ---------------- GET LATEST SENSOR DATA ----------------
def get_sensor_data():
    try:
        ref = db.reference("sensor")
        data = ref.get()

        if not data:
            return None

        # Firebase has timestamp-like keys → get latest entry
        latest_key = sorted(data.keys())[-1]
        return data[latest_key]

    except Exception as e:
        st.error(f"Firebase Error: {e}")
        return None

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

# ---------------- CHATBOT ----------------
@st.cache_resource
def load_chatbot():
    return pipeline("text-generation", model="distilgpt2")

chatbot = load_chatbot()

def get_response(user_input):
    result = chatbot(user_input, max_length=100, num_return_sequences=1)
    return result[0]["generated_text"]

# ---------------- SIDEBAR ----------------
page = st.sidebar.selectbox(
    "Choose Page",
    ["🌿 Disease Detection", "🤖 AI Chatbot", "📡 Live Sensor Data"]
)

# =====================================================
# 🌿 DISEASE DETECTION
# =====================================================
if page == "🌿 Disease Detection":

    st.title("🌿 Plant Disease Detection System")

    uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image")

        prediction = model.predict(preprocess(image))

        label = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: {confidence*100:.2f}%")

        if label in disease_info:
            st.subheader("🦠 Cause")
            st.write(disease_info[label]["cause"])

            st.subheader("🛡 Prevention")
            st.write(disease_info[label]["prevention"])

# =====================================================
# 🤖 CHATBOT
# =====================================================
elif page == "🤖 AI Chatbot":

    st.title("🤖 AI Farming Chatbot")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask something...")

    if user_input:

        st.chat_message("user").write(user_input)
        st.session_state.chat.append({"role": "user", "content": user_input})

        with st.spinner("Thinking..."):
            reply = get_response(user_input)

        st.chat_message("assistant").write(reply)
        st.session_state.chat.append({"role": "assistant", "content": reply})

# =====================================================
# 📡 SENSOR DATA
# =====================================================
elif page == "📡 Live Sensor Data":

    st.title("📡 Live Farm Sensor Data")

    data = get_sensor_data()

    if data:
        st.metric("🌡 Temperature", f"{data.get('temp', 0)} °C")
        st.metric("💧 Humidity", f"{data.get('hum', 0)} %")
        st.metric("🌱 Soil Moisture", f"{data.get('soil', 0)}")
    else:
        st.warning("No sensor data found in Firebase")
