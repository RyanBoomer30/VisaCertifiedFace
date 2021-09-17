import cv2
import tensorflow as tf
import os
import numpy as np

processed_image_folder = "Data"
Categories = ["Good", "Bad"]

Img_size = 50
testing_data = []

for category in Categories:
    path = os.path.join(processed_image_folder, "Testing", category)
    class_num = Categories.index(category)
    for img in os.listdir(path):
        try:
            print("Loading test image", img)
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (Img_size, Img_size))
            testing_data.append([new_array, class_num])
        except Exception as e:
            pass

x_test = []
y_test = []

for features, label in testing_data:
    x_test.append(features)
    y_test.append(label)

x_test = np.array(x_test).reshape(-1, Img_size, Img_size, 1)
x_test = x_test/255
# x_test = np.array(x_test)
y_test = np.array(y_test)

model = tf.keras.models.load_model(processed_image_folder)

result = model.evaluate(x_test, y_test, batch_size=32 )
print(result)
