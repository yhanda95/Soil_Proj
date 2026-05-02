import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import pandas as pd
import plotly.express as px
import firebase_admin
from firebase_admin import credentials, db
import google.generativeai as genai
from streamlit_autorefresh import st_autorefresh

# ================= CONFIG =================
st.set_page_config(page_title="Smart Agriculture", layout="wide")
st_autorefresh(interval=3000, key="refresh")

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.h5")

model = load_model()

# ================= LOAD FILES =================
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

with open("disease_info.json", "r") as f:
    disease_info = json.load(f)

class_names = list(class_indices.keys())

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
    "🤖 Chatbot"
])

# =====================================================
# 🌿 DISEASE DETECTION
# =====================================================
if page == "🌿 Disease Detection":

    st.subheader("Upload Leaf Image")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, use_column_width=True)

        processed = preprocess(img)
        pred = model.predict(processed)

        predicted_class = class_names[np.argmax(pred)]
        confidence = np.max(pred) * 100

        st.success(f"{predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")

        if predicted_class in disease_info:
            st.subheader("Cause")
            st.write(disease_info[predicted_class]["cause"])

            st.subheader("Prevention")
            st.write(disease_info[predicted_class]["prevention"])

# =====================================================
# 📊 LIVE DATA
# =====================================================
elif page == "📊 Live Data":

    st.subheader("Real-Time Sensor Data")

    data = ref.get()

    if data:
        latest = list(data.values())[-1]

        soil = latest['soil']
        temp = latest['temp']
        hum = latest['hum']

        c1, c2, c3 = st.columns(3)
        c1.metric("🌱 Soil", soil)
        c2.metric("🌡 Temp", temp)
        c3.metric("💧 Humidity", hum)

        # Smart alerts
        st.markdown("### ⚠ Condition Analysis")

        if soil < 1500:
            st.error("Soil is too dry → Irrigation needed")
        elif soil < 3000:
            st.warning("Soil moisture moderate")
        else:
            st.success("Soil moisture good")

        if hum > 80 and 20 < temp < 30:
            st.error("High risk of fungal disease")

# =====================================================
# 📈 ANALYTICS
# =====================================================
elif page == "📈 Analytics":

    st.subheader("Sensor Trends")

    data = ref.get()

    if data:
        df = pd.DataFrame(list(data.values()))
        df = df.tail(50)

        fig1 = px.line(df, y=["soil", "temp", "hum"], title="Sensor Trends")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.area(df, y="soil", title="Soil Moisture")
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.histogram(df, x="hum", title="Humidity Distribution")
        st.plotly_chart(fig3, use_container_width=True)

# =====================================================
# 🤖 CHATBOT
# =====================================================
elif page == "🤖 Chatbot":

    st.subheader("AI Farming Assistant")

    user_input = st.text_area("Ask your question")

    data = ref.get()

    if data:
        latest = list(data.values())[-1]
        soil = latest['soil']
        temp = latest['temp']
        hum = latest['hum']

    if user_input:
        with st.spinner("Thinking..."):
            prompt = f"""
            Soil: {soil}
            Temperature: {temp}
            Humidity: {hum}

            Question: {user_input}

            Give practical farming advice.
            """

            response = chat_model.generate_content(prompt)
            st.success(response.text)
