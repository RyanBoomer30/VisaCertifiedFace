# Import
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import shutil
import os
import pathlib
import cv2
import random
import pickle

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

# File split
poses = []
anno = []
good = dict()
bad = dict()

with open('CelebAMask-HQ-pose-anno.txt') as p:
    for line in p:
        data = line.strip().split('.jpg ', 1)
        try:
            yaw, pitch, roll = (item.strip() for item in data[1].split(' ', 2))
            yaw = float(yaw)
            pitch = float(pitch)
            roll = float(roll)
            if yaw > -10 and yaw < 10 and pitch > -20 and pitch < 5 and roll > -5 and roll < 5:
                good[data[0]] = {"data": [yaw, pitch, roll, -1], "status": "1"}
            else:
                bad[data[0]] = {"data": [yaw, pitch, roll, -1], "status": "0"}
        except IndexError:
            continue

with open('CelebAMask-HQ-attribute-anno.txt') as p:
    for line in p:
        data = line.strip().split('.jpg  ', 1)
        try:
            Five_Clock_Shadow, Arched_Eyebrows, Attractive, Bags_Under_Eyes, Bald, Bangs, Big_Lips, Big_Nose, Black_Hair, Blond_Hair, Blurry, Brown_Hair, Bushy_Eyebrows, Chubby, Double_Chin, Eyeglasses, Goatee, Gray_Hair, Heavy_Makeup, High_Cheekbones, Male, Mouth_Slightly_Open, Mustache, Narrow_Eyes, No_Beard, Oval_Face, Pale_Skin, Pointy_Nose, Receding_Hairline, Rosy_Cheeks, Sideburns, Smiling, Straight_Hair, Wavy_Hair, Wearing_Earrings, Wearing_Hat, Wearing_Lipstick, Wearing_Necklace, Wearing_Necktie, Young = (item.strip() for item in data[1].split(' ', 39))
            if int(Eyeglasses) == 1 and data[0] in good:
                bad[data[0]] = good[data[0]]
                bad[data[0]]["status"] = "0"
                bad[data[0]]["data"][-1] = 1
                del (good[data[0]])
        except IndexError:
            continue

# Change this into your good file url
good_destination = 'E:\Internship\CelebAMask-HQ\Good'
for i in good.keys():
    dest = shutil.move('E:\Internship\CelebAMask-HQ\CelebA-HQ-img\{}.jpg'.format(i), good_destination) 

# Change this into your bad file url
bad_destination = 'E:\Internship\CelebAMask-HQ\Bad'
for i in bad.keys():
    dest = shutil.move('E:\Internship\CelebAMask-HQ\CelebA-HQ-img\{}.jpg'.format(i), bad_destination) 

# Prepare data
# Change this into your training folder
Datadir = "E:\Internship\CelebAMask-HQ\Training"
Categories = ['Good', 'Bad']

Img_size = 50

training_data = []

for category in Categories:
    path = os.path.join(Datadir, category)
    class_num = Categories.index(category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (Img_size, Img_size))
            training_data.append([new_array, class_num])
        except Exception as e:
            pass

random.shuffle(training_data)

x = []
y = []

for features, label in training_data:
    x.append(features)
    y.append(label)

x = np.array(x).reshape(-1, Img_size, Img_size, 1)
pickle_out = open("x.pickle", "wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

# Train model
x = pickle.load(open("x.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

x = x/255

x = np.array(x)
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

model.fit(x, y, batch_size=32, epochs=20, validation_split=0.2)

# Change this to your model save url
model.save('E:\Internship\CelebAMask-HQ\Training')