import cv2
import tensorflow as tf
import os
import numpy as np
import shutil
import uuid

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

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
print("Haar model file: ", haar_model)

# load file and crop the face
def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    cascade = cv2.CascadeClassifier(haar_model)
    face = cascade.detectMultiScale(img_array, 1.3, 5)

    height, width = img_array.shape
    if len(face) == 0:
        print("No face at", filepath)
        return img_array
    else:
        x, y, w, h = face[0]
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        ext_ratio = 5
        start_y = int(y-(w/ext_ratio)) if int(y-(w/ext_ratio)) > 0 else 0
        end_y = int(y+h+(w/ext_ratio)) if int(y+h+(w/ext_ratio)) < height else height
        start_x = int(x-(w/ext_ratio)) if int(x-(w/ext_ratio)) > 0 else 0
        end_x = int(x+w+(w/ext_ratio)) if int(x+w+(w/ext_ratio)) < width else width
        
        crop_img = img_array[start_y:end_y, start_x:end_x]

        return crop_img

# Load model
model = tf.keras.models.load_model(processed_image_folder)

# Generate predicted result into Good and Bad folders
for i in os.listdir(test_image_folder):
    test_file = os.path.join(test_image_folder, i)

    test_image = prepare(test_file)

    normalized_image = cv2.resize(test_image, (Img_size, Img_size))
    # cv2.imshow("picture", normalized_image)
    # cv2.waitKey(0)
    normalized_image = np.array(normalized_image).reshape(-1, Img_size, Img_size, 1) / 255

    score = model.predict(normalized_image)[0][0]
    if score <= 0.58367187:
        # dest = shutil.copy(test_file, goodResult_destination)
        filename = os.path.join(goodResult_destination, str(score) + ".jpg")
        print(filename)
        res = cv2.imwrite(filename, test_image)
        print("File copied: ", score, res)

    else:
        # dest = shutil.copy(test_file, badResult_destination)
        filename = os.path.join(badResult_destination, str(score) + ".jpg")
        print(filename)
        res = cv2.imwrite(filename, test_image)
        print("File copied: ", score, res)
