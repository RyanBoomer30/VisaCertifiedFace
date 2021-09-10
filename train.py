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

# File
main_folder = "E:\Internship\CelebAMask-HQ\Training"

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
model.save(main_folder)