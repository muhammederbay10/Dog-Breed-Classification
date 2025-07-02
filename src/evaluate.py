import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

from config import Config
from src.data_pipeline import create_data_batches  
from scripts.prepare_data import load_and_split_data  

def evaluate_model():
    # Load validation data directly
    split_data, class_names = load_and_split_data()
    _, x_val, _, y_val = split_data
    valid_data = create_data_batches(x_val, y_val, valid_data=True)
    
    # Load saved model
    model = tf.keras.models.load_model(
        Config.MODEL_SAVE_PATH,
        custom_objects={'KerasLayer': hub.KerasLayer})
    
    # Evaluate
    # Predict in smaller batches
    y_probs = []
    for batch in valid_data:
        y_probs.append(model.predict(batch[0], verbose=0))
    y_probs = np.concatenate(y_probs, axis=0)
    y_preds = y_probs.argmax(axis=1)
    y_true = np.concatenate([y for x, y in valid_data], axis=0)

    # If y_true is one-hot encoded, convert to integers
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = y_true.argmax(axis=1)
    
    print(classification_report(y_true, y_preds, target_names=class_names))

    # Confusion Matrix (optional)
    cm = confusion_matrix(y_true, y_preds)
    plt.figure(figsize=(40, 35))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.show()

if __name__ == "__main__":
    evaluate_model()