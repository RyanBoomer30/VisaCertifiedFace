# Import
import numpy as np
import cv2
import random
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

processed_image_folder = "Data"

Categories = ['Good', 'Bad']

Img_size = 50
training_data = []

print("Preparing Data")
for category in Categories:
    path = os.path.join(processed_image_folder, "Training", category)
    class_num = Categories.index(category)
    for img in os.listdir(path):
        try:
            print("Loading train image", img)
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (Img_size, Img_size))
            training_data.append([new_array, class_num])
        except Exception as e:
            pass

random.shuffle(training_data)

print("Process training data")
x = []
y = []

for features, label in training_data:
    x.append(features)
    y.append(label)

x = np.array(x).reshape(-1, Img_size, Img_size, 1)
x = x/255
# x = np.array(x)
y = np.array(y)

model = Sequential()

# Layer 1
model.add(Conv2D(64, (3,3), input_shape=x.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# Layer 2
model.add(Conv2D(64, (3,3), input_shape=x.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# Layer 3
model.add(Flatten())
model.add(Dense(64))

# Output
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x, y, batch_size=32, epochs=10, validation_split=0.2)

# Change this to your model save url
model.save(processed_image_folder)
