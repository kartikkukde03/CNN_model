import numpy as np
import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model from your specified path
model_path = r"C:\\MY DATA\\6th sem\\BMDL\\TAE-1\\model\\mnist_cnn.h5"
try:
    model = load_model(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Initialize Tkinter
root = tk.Tk()
root.title("MNIST Digit Recognizer")

# Canvas settings
canvas_size = 280  # 280x280 pixels (10x MNIST size for better drawing)
canvas = Canvas(root, width=canvas_size, height=canvas_size, bg="black")
canvas.pack()

# Create a white image for drawing
image = Image.new("L", (canvas_size, canvas_size), "black")
draw = ImageDraw.Draw(image)

# Function to draw on the canvas
def draw_digit(event):
    x, y = event.x, event.y
    radius = 12  # Brush size
    canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="white", outline="white")
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill="white")

# Function to clear the canvas
def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_size, canvas_size], fill="black")

# Function to predict the digit
def predict_digit():
    # Resize to 28x28 with antialiasing
    img_resized = image.resize((28, 28), Image.LANCZOS).convert("L")
    
    # Convert to numpy array & Normalize
    img_array = np.array(img_resized) / 255.0
    
    # Center the digit by thresholding & bounding box
    img_array = np.where(img_array > 0.2, 1.0, 0.0)  # Make the digit sharper
    from scipy.ndimage import center_of_mass
    cy, cx = center_of_mass(img_array)  # Get digit's center
    shift_y, shift_x = int(14 - cy), int(14 - cx)  # Compute shift
    img_array = np.roll(img_array, shift=(shift_y, shift_x), axis=(0, 1))  # Center it
    
    # Reshape for CNN
    img_array = img_array.reshape(1, 28, 28, 1)

    # Get prediction
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    # Show result
    result_label.config(text=f"Prediction: {predicted_digit}", font=("Arial", 24))


# Bind mouse movement to drawing function
canvas.bind("<B1-Motion>", draw_digit)

# Buttons
btn_predict = tk.Button(root, text="Predict", command=predict_digit, font=("Arial", 16))
btn_predict.pack(pady=10)

btn_clear = tk.Button(root, text="Clear", command=clear_canvas, font=("Arial", 16))
btn_clear.pack()

# Prediction result label
result_label = tk.Label(root, text="Draw a digit and click Predict!", font=("Arial", 18))
result_label.pack()

# Run the Tkinter event loop
root.mainloop()
