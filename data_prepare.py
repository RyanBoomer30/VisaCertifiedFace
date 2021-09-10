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

# File
main_folder = "E:\Internship\CelebAMask-HQ\Training"
image_folder = "E:\Internship\CelebAMask-HQ\CelebA-HQ-img"
test_destination = "E:\Internship\CelebAMask-HQ\Training\Test"
good_destination = 'E:\Internship\CelebAMask-HQ\Training\Good'
bad_destination = 'E:\Internship\CelebAMask-HQ\Training\Bad'


# Create a test file
print("Creating Test Folder")
for i in range(6001):
    dest = shutil.move('{}\{}.jpg'.format(image_folder,i), test_destination) 

with open('CelebAMask-HQ-pose-anno.txt') as p:
    for line in p:
        data = line.strip().split('.jpg ', 1)
        try:
            check = float(data[0])
            if (check>6000):
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
        except ValueError:
            continue

with open('CelebAMask-HQ-attribute-anno.txt') as p:
    for line in p:
        data = line.strip().split('.jpg  ', 1)
        try:
            check = float(data[0])
            if (check>6000):
                Five_Clock_Shadow, Arched_Eyebrows, Attractive, Bags_Under_Eyes, Bald, Bangs, Big_Lips, Big_Nose, Black_Hair, Blond_Hair, Blurry, Brown_Hair, Bushy_Eyebrows, Chubby, Double_Chin, Eyeglasses, Goatee, Gray_Hair, Heavy_Makeup, High_Cheekbones, Male, Mouth_Slightly_Open, Mustache, Narrow_Eyes, No_Beard, Oval_Face, Pale_Skin, Pointy_Nose, Receding_Hairline, Rosy_Cheeks, Sideburns, Smiling, Straight_Hair, Wavy_Hair, Wearing_Earrings, Wearing_Hat, Wearing_Lipstick, Wearing_Necklace, Wearing_Necktie, Young = (item.strip() for item in data[1].split(' ', 39))
                if int(Eyeglasses) == 1 and data[0] in good:
                    bad[data[0]] = good[data[0]]
                    bad[data[0]]["status"] = "0"
                    bad[data[0]]["data"][-1] = 1
                    del (good[data[0]])
        except IndexError:
            continue
        except ValueError:
            continue

# Change this into your good file url
print("Creating Good Folder")
for i in good.keys():
    dest = shutil.move('{}\{}.jpg'.format(image_folder,i), good_destination) 

# Change this into your bad file url
print("Creating Bad Folder")
for i in bad.keys():
    dest = shutil.move('{}\{}.jpg'.format(image_folder,i), bad_destination) 

# Prepare data
print("Preparing Data")
Datadir = main_folder
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

test_dataset = []

for img in os.listdir(test_destination):
    destination = test_destination + "/" + img
    img_array = cv2.imread(destination, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (Img_size, Img_size))
    test_dataset.append(new_array)

test_dataset = np.array(test_dataset).reshape(-1, Img_size, Img_size, 1)

pickle_out = open("test.pickle", "wb")
pickle.dump(test_dataset, pickle_out)
pickle_out.close()