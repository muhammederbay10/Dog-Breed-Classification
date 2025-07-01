class Config:
    RAW_DATA_DIR = "data/raw"
    PROC_DATA_DIR = "data/processed"
    ZIP_FILENAME = "dog-breed-identification.zip"
    DATA_URL = "https://www.kaggle.com/c/dog-breed-identification/data"
    IMG_SIZE = 224
    NUM_CLASSES = 120
    UNIQUE_BREEDS = None
    BATCH_SIZE = 32
    INPUT_SHAPE = [None, 224, 224, 3]
    OUTPUT_SHAPE = 120
    NUM_EPOCHS = 100
    
