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

# File split
poses = []
anno = []
good = dict()
bad = dict()
goodTest = dict()
badTest = dict()

# File
main_folder = "E:\Internship\CelebAMask-HQ\Training"
image_folder = "E:\Internship\CelebAMask-HQ\CelebA-HQ-img"
goodTest_destination = "E:\Internship\CelebAMask-HQ\Training\GoodTest"
badTest_destination = "E:\Internship\CelebAMask-HQ\Training\BadTest"
good_destination = 'E:\Internship\CelebAMask-HQ\Training\Good'
bad_destination = 'E:\Internship\CelebAMask-HQ\Training\Bad'

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
print("Creating Good Folder")
for i in good.keys():
    dest = shutil.move('{}\{}.jpg'.format(image_folder,i), good_destination) 

# Change this into your bad file url
print("Creating Bad Folder")
for i in bad.keys():
    dest = shutil.move('{}\{}.jpg'.format(image_folder,i), bad_destination)

# Create good test file
count = 0
for i in os.listdir(good_destination):
    if count < 3000:
        dest = shutil.move('{}\{}'.format(good_destination,i), goodTest_destination)
        count+=1
    else:
        break

# Create bad test file
count = 0
for i in os.listdir(bad_destination):
    if count < 3000:
        dest = shutil.move('{}\{}'.format(bad_destination,i), badTest_destination)
        count+=1
    else:
        break

# Prepare data
print("Preparing Data")
Categories = ['Good', 'Bad']

Img_size = 50

training_data = []
testing_data = []

for category in Categories:
    path = os.path.join(main_folder, category)
    class_num = Categories.index(category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (Img_size, Img_size))
            training_data.append([new_array, class_num])
        except Exception as e:
            pass

for category in Categories:
    if category == 'Good':
        path = goodTest_destination
    elif category == 'Bad':
        path = badTest_destination
    class_num = Categories.index(category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (Img_size, Img_size))
            testing_data.append([new_array, class_num])
        except Exception as e:
            pass

random.shuffle(training_data)

print("Saving training")
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

print("Saving testing")
x_test = []
y_test = []

for features, label in testing_data:
    x_test.append(features)
    y_test.append(label)

x_test = np.array(x_test).reshape(-1, Img_size, Img_size, 1)
pickle_out = open("x_test.pickle", "wb")
pickle.dump(x_test, pickle_out)
pickle_out.close()

pickle_out = open("y_test.pickle", "wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()