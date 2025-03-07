import streamlit as st
import tensorflow as tf
import numpy as np
import urllib.request
import os
from PIL import Image

# Set paths
MODEL_PATH = "mnist_cnn.h5"
MODEL_URL = "https://github.com/kartikkukde03/CNN_model/blob/main/model/mnist_cnn.h5"  # Replace with your actual GitHub raw link

# 🔹 Ensure model file exists, else download it
if not os.path.exists(MODEL_PATH):
    st.warning("Model file not found! Downloading from GitHub...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.success("Model downloaded successfully!")
    except Exception as e:
        st.error(f"Failed to download model: {e}")
        st.stop()

# 🔹 Load the trained CNN model
model = tf.keras.models.load_model(MODEL_PATH)

# Streamlit UI
st.title("🖥️ MNIST Digit Classifier")
st.write("Upload a hand-drawn digit image (28x28 pixels), and the model will predict it!")

# Upload Image
uploaded_file = st.file_uploader("Choose a digit image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    img_array = np.array(image).reshape(1, 28, 28, 1) / 255.0  # Normalize

    # Show the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=False)

    # 🔹 Make prediction
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    # Display result
    st.subheader(f"🧠 Model Prediction: **{predicted_digit}**")
