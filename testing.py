import cv2
import tensorflow as tf


CATEGORIES = ["S11", "S21","S31","S41","S51"]


def prepare(filepath):
    IMG_SIZE = 50  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # return the image with shaping that TF wants.


model = tf.keras.models.load_model(r"C:\Users\gaell\OneDrive\Documents\SCE- Schepens\Official project\TrianData\64x3-CNN.model")

prediction = model.predict([prepare(r"C:\Users\gaell\OneDrive\Documents\SCE- Schepens\Official project\Students\Student1.jpg")])
print (prediction)
print(CATEGORIES[int(prediction[0][0])])
