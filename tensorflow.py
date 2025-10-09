import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt

model = MobileNetV2(weights='imagenet')

img_path = "D:/pc.jpg"
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = preprocess_input(img_array)

predictions = model.predict(np.expand_dims(img_array, axis=0))
decoded_predictions = decode_predictions(predictions, top=3)[0]

for _, label, probability in decoded_predictions:
    print(f"{label}: {probability:.2%}")
