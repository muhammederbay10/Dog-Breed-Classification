import tensorflow as tf 
import matplotlib.pyplot as plt
from src.data_pipeline import process_image, get_image_label, create_data_batches
from scripts.prepare_data import load_and_split_data
from config import Config

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

