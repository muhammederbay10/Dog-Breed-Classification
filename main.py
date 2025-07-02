import tensorflow_hub as hub
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from src.data_pipeline import create_data_batches
from src.predict import load_and_preprocess_image, predict_breed
from scripts.prepare_data import load_and_split_data
from keras.models import load_model
from config import Config


import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib
import sklearn
import seaborn

print(f"TensorFlow: {tf.__version__}")
print(f"TensorFlow Hub: {hub.__version__}")
print(f"Numpy: {np.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print(f"Seaborn: {seaborn.__version__}")


# Get split data and class names
split_data, class_names = load_and_split_data()
x_train, x_val, y_train, y_val = split_data

# Create batches
train_data = create_data_batches(x_train, y_train)
valid_data = create_data_batches(x_val, y_val, valid_data=True)

Config.UNIQUE_BREEDS = class_names


def show_25_images(images, labels):
    """
    Display 25 images and their labels from the data batch
    """
    # Setup the figure
    plt.figure(figsize=(10, 10))

    # Loop through 25 images
    for i in range(25):
        # Create subplot (5rows, 5columns)
        plt.subplot(5,5, i+1)
        # Plot the image
        plt.imshow(images[i])
        # Set the title of the subplot to the label
        plt.title(Config.UNIQUE_BREEDS[labels[i].argmax()])
        # Turn off the axis
        plt.axis("off")

train_images, train_labels = next(train_data.as_numpy_iterator())
show_25_images(train_images, train_labels)
plt.show()

# Load the model
model = load_model(Config.MODEL_SAVE_PATH, custom_objects={'KerasLayer': hub.KerasLayer})

# Load and preprocess the image
img_path = "Data/processed/test/loka.jpg"
img_array = load_and_preprocess_image(img_path)

# Predict 
breed, confidence = predict_breed(model, img_array, Config.UNIQUE_BREEDS)
print(f"Predicted breed: {breed} ({confidence*100:.2f}% confidence)")

# Load the original image (for visualization)
img = mpimg.imread(img_path)

# Plot the image
plt.imshow(img)
plt.axis("off")

# Title with breed and confidence
plt.title(breed, fontsize=14, color="green")

# Show the image
plt.show()

