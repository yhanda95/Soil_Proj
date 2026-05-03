import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

from transformers import pipeline

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Plant Disease Detection 🌿", layout="centered")

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

# ---------------- PREPROCESS ----------------
def preprocess(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image

# ---------------- SIMPLE AI CHATBOT ----------------
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

# ---------------- SIDEBAR NAVIGATION ----------------
page = st.sidebar.selectbox("Choose Page", ["🌿 Disease Detection", "🤖 AI Chatbot"])

# =====================================================
# 🌿 PAGE 1 - DISEASE DETECTION
# =====================================================
if page == "🌿 Disease Detection":

    st.title("🌿 Plant Disease Detection System")
    st.write("Upload a leaf image to detect disease and get treatment info.")

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
        else:
            st.warning("No disease info available.")

# =====================================================
# 🤖 PAGE 2 - CHATBOT
# =====================================================
elif page == "🤖 AI Chatbot":

    st.title("🤖 AI Farming Chatbot")
    st.write("Ask anything about plants, farming, or diseases.")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    # show chat history
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask something...")

    if user_input:

        # user message
        st.chat_message("user").write(user_input)
        st.session_state.chat.append({"role": "user", "content": user_input})

        # bot response
        with st.spinner("Thinking..."):
            reply = get_response(user_input)

        st.chat_message("assistant").write(reply)
        st.session_state.chat.append({"role": "assistant", "content": reply})
