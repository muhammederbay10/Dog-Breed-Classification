import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

def load_and_split_data():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(BASE_DIR, 'Data', 'processed', 'labels.csv')
    image_dir = os.path.join(BASE_DIR, 'Data', 'processed', 'train')
    
    labels_df = pd.read_csv(csv_path)
    # Create full image paths
    x = [os.path.join(image_dir, f"{img_id}.jpg") for img_id in labels_df["id"]]  
    y = labels_df["breed"]

    # Encode String Labels into int
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return train_test_split(x, y_encoded, test_size=0.2, random_state=42), label_encoder.classes_