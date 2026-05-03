import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Plant Disease Detection")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras")

model = load_model()

# TEMP (replace with real names later)
class_names = ["Healthy", "Diseased"]

def preprocess(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image

st.title("🌿 Plant Disease Detection System")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed = preprocess(image)
    prediction = model.predict(processed)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence*100:.2f}%")
