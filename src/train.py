import numpy as np
import tensorflow as tf
import os
import datetime
from model import create_model
from main import train_data, valid_data
from config import Config

def create_tensorboard_callbacks():
    """
    creats a tensorboard callbacks which is able to save logs into directory and pass it to our model's 'fit()' function.
    """
    # Creats a log file
    logdir = os.path.join("/logs",
                          # Make it so the logs get tracled whenever we run the expirement
                          datetime.datetime.now().strftime("%y%m%d-H%M%S"))
    return tf.keras.callbacks.TensorBoard(logdir)

# Create earlystopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                  patience=3)

def train_model():
    """
    Trains a given model and returns the trained version.
    """
    # Create Model
    model = create_model()

    # Create a tensorborad session everytime we train a model
    tensorboard = create_tensorboard_callbacks()

    # Fit the model to the data passing callbacks we created
    model.fit(x=train_data,
              epochs=Config.NUM_EPOCHS,
              validation_data=valid_data,
              validation_freq=1,
              callbacks=[tensorboard, early_stopping])
    return model

model = train_model()