import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Load trained model with error handling
try:
    model = load_model("C:\\MY DATA\\6th sem\\BMDL\\TAE-1\\model\\mnist_cnn.h5")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Load MNIST test dataset
(_, _), (x_test, y_test) = mnist.load_data()

# Normalize and reshape for CNN
x_test = x_test / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

# Pick a random image from the test set
index = np.random.randint(0, len(x_test))
test_image = x_test[index]
true_label = y_test[index]

# Display the test image
plt.imshow(test_image.squeeze(), cmap="gray")
plt.title(f"Actual Label: {true_label}")
plt.axis("off")
plt.show()

# Predict using the model
prediction = model.predict(test_image.reshape(1, 28, 28, 1))
predicted_label = np.argmax(prediction)

# Print prediction results
print(f"🧠 Model Prediction: {predicted_label}")
print(f"✅ Actual Label: {true_label}")
