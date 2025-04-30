from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf, os

IMG_SIZE = (224, 224, 3)
NUM_CLASSES = int(os.getenv("NUM_CLASSES", "4"))   # override if needed
MODEL_PATH  = os.getenv("MODEL_PATH", "models/classifier/classifier.weights.h5")

def build_model():
    model = Sequential([
        Conv2D(32, 3, activation='relu', input_shape=IMG_SIZE),
        MaxPooling2D(2),
        Conv2D(64, 3, activation='relu'),
        MaxPooling2D(2),
        Conv2D(128, 3, activation='relu'),
        MaxPooling2D(2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile()  # not needed for inference but loads weights
    return model

def load_model():
    model = build_model()
    model.load_weights(MODEL_PATH)
    return model

MODEL = load_model()
