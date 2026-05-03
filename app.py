import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import google.generativeai as genai
from dotenv import load_dotenv

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Plant Disease AI 🌿", layout="centered")

# ---------------- LOAD ENV ----------------
load_dotenv()
genai.configure(api_key=os.getenv("AIzaSyBN4g03--SHgVne8N3cnuhxKF5r25qpYac"))

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras")

model = load_model()

# ---------------- LOAD DISEASE INFO ----------------
with open("disease_info.json", "r") as f:
    disease_info = json.load(f)

# ---------------- CLASS NAMES ----------------
class_names = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_healthy"
]

# ---------------- GEMINI FUNCTION ----------------
def query_gemini(question, disease):
    try:
        model_ai = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""
You are an expert agricultural assistant.

Detected disease: {disease}

User question: {question}

Give simple, practical farming advice including:
- treatment
- prevention
- fertilizer suggestions if needed
"""

        response = model_ai.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# ---------------- PREPROCESS ----------------
def preprocess(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image

# ---------------- UI ----------------
st.title("🌿 Plant Disease Detection AI")
st.write("Upload a leaf image and get disease prediction + AI advice")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

# ---------------- PREDICTION ----------------
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image")

    processed = preprocess(image)
    prediction = model.predict(processed)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    st.subheader(f"🧠 Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence*100:.2f}%")

    # ---------------- DISEASE INFO ----------------
    if predicted_class in disease_info:
        st.subheader("🦠 Cause")
        st.write(disease_info[predicted_class]["cause"])

        st.subheader("🛡 Prevention")
        st.write(disease_info[predicted_class]["prevention"])

    # ---------------- CHATBOT ----------------
    st.divider()
    st.subheader("🤖 Ask AI Farming Assistant")

    user_question = st.text_input("Ask about treatment, fertilizer, prevention, etc.")

    if user_question:
        answer = query_gemini(user_question, predicted_class)
        st.success(answer)
