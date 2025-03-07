import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Load trained model
MODEL_PATH = r"mnist_cnn.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Preprocessing function
def preprocess_image(img):
    img = img.convert("L")  # Convert to grayscale
    img = img.resize((28, 28), Image.LANCZOS)
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

# Streamlit UI
st.title("🖍️ MNIST Handwritten Digit Recognition")
st.write("Draw a digit below and get predictions!")

# Canvas for drawing
from streamlit_drawable_canvas import st_canvas
canvas = st_canvas(stroke_width=10, background_color="white", height=300, width=300)

if st.button("Predict"):
    if canvas.image_data is not None:
        # Convert to PIL image
        img = Image.fromarray((canvas.image_data[:, :, :3]).astype("uint8"))
        processed_img = preprocess_image(img)

        # Prediction
        prediction = model.predict(processed_img)
        digit = np.argmax(prediction)

        # Display result
        st.success(f"🎉 Predicted Digit: {digit}")
