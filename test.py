import cv2
import tensorflow as tf
import os
import numpy as np
import pickle

Categories = ["Good", "Bad"]

main_folder = "E:\Internship\CelebAMask-HQ\Training"
test_destination = "E:\Internship\CelebAMask-HQ\Training\Test"

test_dataset = pickle.load(open("test.pickle", "rb"))

model = tf.keras.models.load_model(main_folder)

result = model.evaluate(test_dataset)
print(dict(zip(model.metrics_names, result)))