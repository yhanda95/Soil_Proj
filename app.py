import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import google.generativeai as genai

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Smart Agriculture AI 🌿🤖", layout="centered")

# ---------------- GEMINI ----------------
genai.configure(api_key="YOUR_GEMINI_API_KEY")

def get_gemini_response(prompt):
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

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

# ---------------- SIDEBAR NAV ----------------
page = st.sidebar.selectbox("Select Page", ["🌿 Disease Detection", "🤖 AI Chatbot"])

# =========================================================
# 🌿 PAGE 1 - DISEASE DETECTION
# =========================================================
if page == "🌿 Disease Detection":

    st.title("🌿 Plant Disease Detection System")

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

        # Disease info
        if predicted_class in disease_info:
            st.subheader("🦠 Cause")
            st.write(disease_info[predicted_class]["cause"])

            st.subheader("🛡 Prevention")
            st.write(disease_info[predicted_class]["prevention"])

        # Quick AI help
        st.divider()
        st.subheader("🤖 Ask AI About This Disease")

        question = st.text_input("Ask something (treatment, fertilizer, etc.)")

        if question:
            with st.spinner("Thinking..."):
                prompt = f"""
                Disease: {predicted_class}
                Question: {question}
                Give simple farming advice in 6-8 lines.
                """
                answer = get_gemini_response(prompt)
                st.success(answer)

# =========================================================
# 🤖 PAGE 2 - CHATBOT
# =========================================================
elif page == "🤖 AI Chatbot":

    st.title("🤖 Gemini AI Chatbot")
    st.write("Ask anything — farming, tech, science, general knowledge")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    # show history
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Type your message...")

    if user_input:

        st.chat_message("user").write(user_input)
        st.session_state.chat.append({"role": "user", "content": user_input})

        with st.spinner("Thinking..."):
            reply = get_gemini_response(user_input)

        st.chat_message("assistant").write(reply)
        st.session_state.chat.append({"role": "assistant", "content": reply})
