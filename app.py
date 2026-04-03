import streamlit as st
import serial
import time
import numpy as np
from PIL import Image
import tensorflow as tf

# -------------------------------
# 🔌 SERIAL CONNECTION (CHANGE PORT)
# -------------------------------
SERIAL_PORT = '/dev/cu.usbserial-A5069RR4'   # Windows: COM3 / Mac: '/dev/cu.usbmodemXXXX'
BAUD_RATE = 9600

# -------------------------------
# 🤖 LOAD MODEL
# -------------------------------
model = tf.keras.models.load_model("model.h5")

class_names = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___healthy"
]

# -------------------------------
# 📸 IMAGE PREPROCESSING
# -------------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -------------------------------
# 🔌 READ SENSOR DATA
# -------------------------------
def get_sensor_data():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        line = ser.readline().decode('utf-8').strip()
        ser.close()

        # Expected format: temp,humidity,soil
        temp, humidity, soil = line.split(',')

        return float(temp), float(humidity), int(soil)

    except:
        return None, None, None

# -------------------------------
# 🧠 RECOMMENDATION ENGINE
# -------------------------------
def get_recommendation(disease, temp, humidity, soil):
    advice = ""

    if disease != "Tomato___healthy":
        advice += "⚠️ Disease detected. Take immediate action.\n"

    if "blight" in disease.lower() and humidity and humidity > 70:
        advice += "💧 High humidity may increase fungal growth.\n"

    if soil is not None:
        if soil < 300:
            advice += "🌱 Soil is dry. Water the plant.\n"
        elif soil > 700:
            advice += "🚫 Soil is too wet. Avoid overwatering.\n"

    if temp and temp > 35:
        advice += "🌡 Temperature too high. Provide shade.\n"

    if advice == "":
        advice = "✅ Plant conditions look good."

    return advice

# -------------------------------
# 🌐 STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="Smart Plant Health System", layout="wide")

st.title("🌿 Smart Plant Disease Detection & Monitoring")

# -------------------------------
# 📊 SENSOR DATA SECTION
# -------------------------------
st.header("📡 Live Sensor Data")

temp, humidity, soil = get_sensor_data()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🌡 Temperature (°C)", temp if temp else "N/A")

with col2:
    st.metric("💧 Humidity (%)", humidity if humidity else "N/A")

with col3:
    st.metric("🌱 Soil Moisture", soil if soil else "N/A")

# -------------------------------
# 📸 IMAGE UPLOAD
# -------------------------------
st.header("📷 Upload Leaf Image")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # -------------------------------
    # 📊 RESULTS
    # -------------------------------
    st.subheader("🧠 Disease Prediction")

    if "healthy" in predicted_class.lower():
        st.success(f"✅ {predicted_class}")
    else:
        st.error(f"🚨 {predicted_class}")

    st.write(f"Confidence: {confidence*100:.2f}%")

    # -------------------------------
    # 💡 RECOMMENDATION
    # -------------------------------
    st.subheader("💡 Smart Recommendation")

    recommendation = get_recommendation(predicted_class, temp, humidity, soil)

    st.info(recommendation)
