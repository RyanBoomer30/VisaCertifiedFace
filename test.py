import cv2
import tensorflow as tf
import os

Categories = ["Good", "Bad"]

def read_file(filepath):
    img_size = 50
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (img_size, img_size))
    return new_array.reshape(-1, img_size, img_size, 1)

# Swap this with your model url
model = tf.keras.models.load_model('E:\Internship\CelebAMask-HQ\Training')

# Swap this with your testing picture url
prediction = model.predict([read_file('E:\Internship\CelebAMask-HQ\Training\Testing\sample1.jpg')])
print( Categories[int(prediction[0][0])])