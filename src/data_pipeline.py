import tensorflow as tf
import matplotlib.pyplot as plt
from config import Config

def process_image(image_path, image_size=Config.IMG_SIZE):
    """
    Takes an image files path as input and turning image into a tensor
    """
    # Read the Image File
    image = tf.io.read_file(image_path)

    # Turn the jpeg image into numerical tensor with 3 colors(Red, Green, Blue)
    image = tf.image.decode_jpeg(image, channels=3)

    # Convert the color value 0-255 into 0-1
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Resize the image to be a simple (224,224)
    image = tf.image.resize(image, size=[image_size, image_size])

    return image

# Create a simple Funtion to return a tuple of image (image, label)
def get_image_label(image_path, label):
    """
    Takes an image file path and the associated label, Preprocess the image and return a tuple of (image, label) 
    """
    image = process_image(image_path)

    label = tf.cast(label, tf.int32)

    # Encoding the labels
    label_onehot = tf.one_hot(label, depth=Config.NUM_CLASSES)

    return image, label_onehot 

# Create a function to turn data into batches 
def create_data_batches(x, y=None, batch_size=Config.BATCH_SIZE, valid_data=False, test_data=False):
    """
    Creats batches of data out of image (x) and label(y) pairs.
    Suffles the data if it is train but doesn't shuffle the data if it is validation data.
    Also accepts test data as input no labels.
    """
    # If the data is test data, we have no labels
    if test_data:
        print("Creating the test data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x))) # Only file path no labels
        data_batch = data.map(process_image).batch(batch_size)
        return data_batch
    
    # If the data is validation data, we don't shuffle the data
    elif valid_data:
        print("Creating the valid data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x), tf.constant(y))) # File Path and labels
        data_batch =data.map(get_image_label).batch(batch_size)
        return data_batch
    
    # If the data is train_data, we shuffle it
    else:
        print("Creating the Train data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x), tf.constant(y))) # File Path and labels
        data = data.shuffle(buffer_size=len(x))
        data_batch =data.map(get_image_label).batch(batch_size)
        return data_batch
    
   
def show_25_images(images, labels):
    """
    Display 25 images and their labels from the data batch
    """
    # Setup the figure
    plt.figure(figsize=(10, 10))

    # Loop through 25 images
    for i in range(25):
        # Create subplot (5rows, 5columns)
        plt.subplot(5+5+i+1)
        # Plot the image
        plt.imshow(images[i])
        # Set the title of the subplot to the label
        plt.title(Config.UNIQUE_BREEDS[labels[i].argmax()])
        # Turn off the axis
        plt.axis("off")

