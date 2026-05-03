import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

import firebase_admin
from firebase_admin import credentials, db

from transformers import pipeline

# =====================================================
# 🔥 CONFIG
# =====================================================
st.set_page_config(
    page_title="Smart Agriculture System 🌿",
    layout="centered"
)

# =====================================================
# 🔥 FIREBASE INIT
# =====================================================
if not firebase_admin._apps:

    cred = credentials.Certificate(dict(st.secrets["firebase"]))

    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://soilproj-eac88-default-rtdb.europe-west1.firebasedatabase.app"
    })

# =====================================================
# 📡 FIREBASE SENSOR DATA
# =====================================================
def get_sensor_data():
    try:
        ref = db.reference("sensor")
        data = ref.get()

        if not data:
            return None

        latest_key = max(data.keys(), key=lambda x: int(x))
        return data[latest_key]

    except Exception as e:
        st.error(f"Firebase Error: {e}")
        return None

# =====================================================
# 🌿 LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras")

model = load_model()

# =====================================================
# 📄 DISEASE INFO
# =====================================================
with open("disease_info.json", "r") as f:
    disease_info = json.load(f)

class_names = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_healthy"
]

# =====================================================
# 🧠 PREPROCESS IMAGE
# =====================================================
def preprocess(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# =====================================================
# 🤖 CHATBOT
# =====================================================
@st.cache_resource
def load_chatbot():
    return pipeline("text-generation", model="distilgpt2")

chatbot = load_chatbot()

def get_response(user_input):
    result = chatbot(
        user_input,
        max_length=100,
        num_return_sequences=1
    )
    return result[0]["generated_text"]

# =====================================================
# 📌 SIDEBAR NAVIGATION
# =====================================================
page = st.sidebar.selectbox(
    "Choose Module",
    ["🌿 Disease Detection", "📡 Sensor Data", "🤖 AI Chatbot"]
)

# =====================================================
# 🌿 DISEASE DETECTION
# =====================================================
if page == "🌿 Disease Detection":

    st.title("🌿 Plant Disease Detection System")

    uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        prediction = model.predict(preprocess(image))

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
# 📡 SENSOR DATA
# =====================================================
elif page == "📡 Sensor Data":

    st.title("📡 Live Sensor Data")

    data = get_sensor_data()

    if data:

        col1, col2, col3 = st.columns(3)

        col1.metric("🌡 Temperature", f"{data.get('temp', 0)} °C")
        col2.metric("💧 Humidity", f"{data.get('hum', 0)} %")
        col3.metric("🌱 Soil Moisture", f"{data.get('soil', 0)}")

    else:
        st.warning("No sensor data found in Firebase")

# =====================================================
# 🤖 CHATBOT PAGE
# =====================================================
elif page == "🤖 AI Chatbot":

    st.title("🤖 Farming AI Chatbot")
    st.write("Ask anything about farming, plants, or diseases.")

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
