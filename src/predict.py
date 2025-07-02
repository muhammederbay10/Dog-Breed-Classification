import tensorflow as tf
import numpy as np
import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from keras.utils import load_img, img_to_array

def load_and_preprocess_image(image_path, target_size=(Config.IMG_SIZE,Config.IMG_SIZE)):
    """
    Loads and preprocess the image to be used for prediction
    """
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0 
    return img_array

def predict_breed(model, img_array, class_names):
    """
    Predicts the breed from a preprocessed image array using the given model
    """
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_names[predicted_class_index]
    confidence = predictions[0][predicted_class_index]
    return predicted_class, confidence
