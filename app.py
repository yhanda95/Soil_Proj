import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

import firebase_admin
from firebase_admin import credentials, db

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart Agriculture System 🌿", layout="wide")

# ---------------- FIREBASE INIT ----------------
if not firebase_admin._apps:

    firebase_config = dict(st.secrets["firebase"])
    firebase_config["private_key"] = firebase_config["private_key"].replace("\\n", "\n")

    cred = credentials.Certificate(firebase_config)

    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://soilproj-eac88-default-rtdb.europe-west1.firebasedatabase.app/"
    })

# ---------------- GET SENSOR DATA ----------------
def get_sensor_data():
    try:
        ref = db.reference("sensor")
        return ref.get()
    except:
        return None

# ---------------- LOAD MODEL ----------------
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
page = st.sidebar.selectbox("Select Module", [
    "🌿 Disease Detection",
    "📡 Live Sensor Data"
])

# =====================================================
# 🌿 DISEASE DETECTION PAGE
# =====================================================
if page == "🌿 Disease Detection":

    st.title("🌿 Plant Disease Detection System")

    uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image")

        prediction = model.predict(preprocess(image))

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
# 📡 SENSOR DATA PAGE
# =====================================================
elif page == "📡 Live Sensor Data":

    st.title("📡 Live Farm Monitoring System")

    data = get_sensor_data()

    if data:
        st.metric("🌡 Temperature", f"{data.get('temperature', 0)} °C")
        st.metric("💧 Humidity", f"{data.get('humidity', 0)} %")
        st.metric("🌱 Soil Moisture", f"{data.get('soil_moisture', 0)} %")
    else:
        st.warning("No sensor data found in Firebase")
