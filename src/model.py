import tensorflow as tf 
import tensorflow_hub as hub
import os 
import sys

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

# Creats a function which builds a keras model
def create_model(Input_shape=Config.INPUT_SHAPE, Output_shape=Config.OUTPUT_SHAPE, Model_url=Config.MODEL_URL):
    """
    Creats a model with keras
    """
    model = tf.keras.Sequential([
        hub.KerasLayer(Model_url), # Layer 1 Input Layer
        tf.keras.layers.Dense(units=Output_shape,
                              activation="softmax") # Layer 2 Output layer
    ])

    # Compile the Model
    model.compile(
        loss = tf.keras.losses.CategoricalCrossentropy(),
        optimizer = tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    # Buikd the model
    model.build(Input_shape)

    return model

model = create_model()
model.summary()