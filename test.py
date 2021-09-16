import cv2
import tensorflow as tf
import os
import numpy as np
import pickle

Categories = ["Good", "Bad"]

main_folder = "E:\Internship\CelebAMask-HQ\Training"
test_destination = "E:\Internship\CelebAMask-HQ\Training\Test"

x_test = pickle.load(open("x_test.pickle", "rb"))
y_test = pickle.load(open("y_test.pickle", "rb"))

x_test = x_test/255

x_test = np.array(x_test)
y_test = np.array(y_test)

model = tf.keras.models.load_model(main_folder)

result = model.evaluate(x_test, y_test, batch_size=32 )
print(result)