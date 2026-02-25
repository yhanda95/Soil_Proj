import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

st.set_page_config(page_title="Plant Disease Detection")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.h5")

model = load_model()

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

with open("disease_info.json", "r") as f:
    disease_info = json.load(f)

class_names = list(class_indices.keys())

def preprocess(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

st.title("🌿 Plant Disease Detection System")
st.write("Upload a plant leaf image to detect disease and view its causes and prevention.")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed_image = preprocess(image)
    prediction = model.predict(processed_image)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.subheader(f"Predicted Disease: {predicted_class}")
    st.write(f"Confidence: {confidence*100:.2f}%")

    if predicted_class in disease_info:
        st.subheader("Cause")
        st.write(disease_info[predicted_class]["cause"])

        st.subheader("Prevention")
        st.write(disease_info[predicted_class]["prevention"])
