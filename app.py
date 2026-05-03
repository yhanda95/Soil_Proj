import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Plant Disease Detection 🌿")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras")

model = load_model()

# ---------------- LOAD DISEASE INFO ----------------
with open("disease_info.json", "r") as f:
    disease_info = json.load(f)

# ---------------- CLASS NAMES ----------------
# IMPORTANT: must match your training folder names
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

# ---------------- UI ----------------
st.title("🌿 Plant Disease Detection System")
st.write("Upload a leaf image to detect disease and get treatment info.")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    # Show image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    # Prediction
    processed = preprocess(image)
    prediction = model.predict(processed)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    # Output
    st.subheader(f"🧠 Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence*100:.2f}%")

    # ---------------- DISEASE INFO ----------------
    if predicted_class in disease_info:
        st.subheader("🦠 Cause")
        st.write(disease_info[predicted_class]["cause"])

        st.subheader("🛡 Prevention")
        st.write(disease_info[predicted_class]["prevention"])
    else:
        st.warning("No disease info available for this prediction.")
