import cv2
import tensorflow as tf
import os
import numpy as np
import shutil


# This folder contains a copy of data from test_image_folder defined just above,
# but in a structure permitting to train and test easily.
# All good and bad folders will be created inside this folder. 
processed_image_folder = "Data"

# This folder is a child of processed_image_folder
# This contains all the images that are being used to predict
test_image_folder = "AnhThe"

# This folder contains a copy of data from test_image_folder defined just above,
# but in a structure permitting to test easily.
# All good and bad folders will be created inside this folder. 
# We can safely delete this folder if we want to run again the script,
result_image_folder = "Result"
goodResult_destination = os.path.join(processed_image_folder, result_image_folder, "Good")
badResult_destination = os.path.join(processed_image_folder, result_image_folder, "Bad")

# If the folder already exist, the function will throw exception, so that you will
# not be able to accidently run this script twice. If you really want to run it 
# again, you have to delete the whole result_image_folder
os.makedirs(goodResult_destination)
os.makedirs(badResult_destination)

Img_size = 50
testing_data = []

# Resize file
def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    height, width = img_array.shape
    crop_img = img_array[0:width, 0:width]
    new_array = cv2.resize(crop_img, (Img_size, Img_size))
    return np.array(new_array).reshape(-1, Img_size, Img_size, 1)

# Load model
model = tf.keras.models.load_model(processed_image_folder)

# Generate predicted result into Good and Bad folders
for i in os.listdir(test_image_folder):
    test_image = os.path.join(test_image_folder, i)
    test_array = prepare(test_image) / 255
    predict = model.predict(test_array)
    if predict <= 0.5:
        dest = shutil.copy(test_image, goodResult_destination)
        print("File copied: ", predict, dest)
    else:
        dest = shutil.copy(test_image, badResult_destination)
        print("File copied: ", predict, dest)
