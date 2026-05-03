import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

import firebase_admin
from firebase_admin import credentials, db

from transformers import pipeline

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart Agriculture System 🌿", layout="wide")

# ---------------- FIREBASE INIT ----------------
if not firebase_admin._apps:

    firebase_config = {
        "type": st.secrets["firebase"]["type"],
        "project_id": st.secrets["firebase"]["project_id"],
        "private_key_id": st.secrets["firebase"]["private_key_id"],
        "private_key": st.secrets["firebase"]["private_key"].replace("\\n", "\n"),
        "client_email": st.secrets["firebase"]["client_email"],
        "client_id": st.secrets["firebase"]["client_id"],
        "auth_uri": st.secrets["firebase"]["auth_uri"],
        "token_uri": st.secrets["firebase"]["token_uri"],
        "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
        "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"]
    }

    cred = credentials.Certificate(firebase_config)

    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://soilproj-eac88-default-rtdb.europe-west1.firebasedatabase.app/"
    })

def get_sensor_data():
    ref = db.reference("sensor")
    return ref.get()

# ---------------- CHATBOT ----------------
@st.cache_resource
def load_chatbot():
    return pipeline("text-generation", model="distilgpt2")

chatbot = load_chatbot()

def ask_ai(question):
    response = chatbot(question, max_length=120, num_return_sequences=1)
    return response[0]["generated_text"]

# ---------------- CNN MODEL ----------------
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

# ---------------- IMAGE PREPROCESS ----------------
def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# ---------------- SIDEBAR ----------------
choice = st.sidebar.radio("Select Module", [
    "🌿 Disease Detection",
    "📡 Live Sensor Data",
    "🤖 AI Chatbot"
])

# =====================================================
# 🌿 DISEASE DETECTION
# =====================================================
if choice == "🌿 Disease Detection":

    st.title("🌿 Plant Disease Detection System")

    file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

    if file:

        img = Image.open(file).convert("RGB")
        st.image(img, caption="Uploaded Image")

        prediction = model.predict(preprocess(img))

        label = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: {confidence*100:.2f}%")

        if label in disease_info:
            st.markdown("### 🦠 Cause")
            st.write(disease_info[label]["cause"])

            st.markdown("### 🛡 Prevention")
            st.write(disease_info[label]["prevention"])

# =====================================================
# 📡 SENSOR DATA (FIREBASE)
# =====================================================
elif choice == "📡 Live Sensor Data":

    st.title("📡 Live Farm Monitoring System")

    data = get_sensor_data()

    if data:
        st.metric("🌡 Temperature", f"{data.get('temperature', 0)} °C")
        st.metric("💧 Humidity", f"{data.get('humidity', 0)} %")
        st.metric("🌱 Soil Moisture", f"{data.get('soil_moisture', 0)} %")
    else:
        st.warning("No data found in Firebase database")

# =====================================================
# 🤖 CHATBOT
# =====================================================
elif choice == "🤖 AI Chatbot":

    st.title("🤖 Farming AI Assistant")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask anything about farming...")

    if user_input:

        st.chat_message("user").write(user_input)
        st.session_state.chat.append({"role": "user", "content": user_input})

        with st.spinner("Thinking..."):
            reply = ask_ai(user_input)

        st.chat_message("assistant").write(reply)
        st.session_state.chat.append({"role": "assistant", "content": reply})
