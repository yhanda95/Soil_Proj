import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(page_title="Smart Plant System 🌿", layout="centered")

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
# 🧠 IMAGE PREPROCESS
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
    result = chatbot(user_input, max_length=120, num_return_sequences=1)
    return result[0]["generated_text"]

# =====================================================
# SIDEBAR
# =====================================================
page = st.sidebar.selectbox(
    "Choose Module",
    ["🌿 Disease Detection", "🤖 AI Chatbot", "📊 Data Analysis"]
)

# =====================================================
# 🌿 DISEASE DETECTION
# =====================================================
if page == "🌿 Disease Detection":

    st.title("🌿 Plant Disease Detection System")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True)

        prediction = model.predict(preprocess(image))

        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        st.subheader(f"Prediction: {predicted_class}")
        st.write(f"Confidence: {confidence*100:.2f}%")

        if predicted_class in disease_info:
            st.write("### Cause")
            st.write(disease_info[predicted_class]["cause"])

            st.write("### Prevention")
            st.write(disease_info[predicted_class]["prevention"])

# =====================================================
# 🤖 CHATBOT
# =====================================================
elif page == "🤖 AI Chatbot":

    st.title("🤖 Farming AI Chatbot")

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
# 📊 DATA ANALYSIS PAGE (UPGRADED)
# =====================================================
elif page == "📊 Data Analysis":

    st.title("📊 Smart IoT Sensor Analytics Dashboard")

    uploaded_file = st.file_uploader("Upload Firebase JSON", type=["json"])

    if uploaded_file:

        import json
        import pandas as pd
        import matplotlib.pyplot as plt

        raw = json.load(uploaded_file)

        # =====================================================
        # 🔥 FIX: YOUR DATA IS INSIDE "sensor_logs"
        # =====================================================
        if "sensor_logs" in raw:
            raw_data = raw["sensor_logs"]
        else:
            raw_data = raw

        # =====================================================
        # CONVERT TO DATAFRAME
        # =====================================================
        records = list(raw_data.values())
        df = pd.DataFrame(records)

        # =====================================================
        # SORT BY TIME
        # =====================================================
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            df = df.sort_values("time")

        st.subheader("📋 Sensor Data Table")
        st.dataframe(df)

        # =====================================================
        # 📌 KPI CARDS (INSIGHTS)
        # =====================================================
        st.subheader("📌 Key Insights")

        col1, col2, col3 = st.columns(3)

        if "temp" in df.columns:
            col1.metric("🌡 Max Temp", f"{df['temp'].max():.2f}")
            col2.metric("🌡 Min Temp", f"{df['temp'].min():.2f}")
            col3.metric("🌡 Avg Temp", f"{df['temp'].mean():.2f}")

        col4, col5, col6 = st.columns(3)

        if "hum" in df.columns:
            col4.metric("💧 Max Hum", f"{df['hum'].max():.2f}")
            col5.metric("💧 Min Hum", f"{df['hum'].min():.2f}")
            col6.metric("💧 Avg Hum", f"{df['hum'].mean():.2f}")

        col7, col8, col9 = st.columns(3)

        if "soil" in df.columns:
            col7.metric("🌱 Max Soil", f"{df['soil'].max():.2f}")
            col8.metric("🌱 Min Soil", f"{df['soil'].min():.2f}")
            col9.metric("🌱 Avg Soil", f"{df['soil'].mean():.2f}")

        # =====================================================
        # 📈 TIME SERIES (SELECTABLE)
        # =====================================================
        st.subheader("📈 Time Series Analysis")

        cols = [c for c in df.columns if c != "time"]
        selected = st.selectbox("Select parameter", cols)

        fig, ax = plt.subplots()
        ax.plot(df["time"], df[selected], marker="o")
        ax.set_title(f"{selected} Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel(selected)
        plt.xticks(rotation=45)

        st.pyplot(fig)

        # =====================================================
        # 📊 MULTI-LINE GRAPH
        # =====================================================
        st.subheader("📊 Multi-Parameter Trend")

        fig2, ax2 = plt.subplots()

        if "temp" in df.columns:
            ax2.plot(df["time"], df["temp"], label="Temperature 🌡")

        if "hum" in df.columns:
            ax2.plot(df["time"], df["hum"], label="Humidity 💧")

        if "soil" in df.columns:
            ax2.plot(df["time"], df["soil"], label="Soil 🌱")

        ax2.set_title("Sensor Trends Over Time")
        ax2.legend()

        plt.xticks(rotation=45)
        st.pyplot(fig2)
