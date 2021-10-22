# Import
from ntpath import join
from sys import path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import shutil
import os
import pickle
import mediapipe as mp
import cv2

# File split
poses = []
anno = []
good = dict()
bad = dict()
goodTest = dict()
badTest = dict()

# Source data, in any case do not modify its content.
# You can also give this value in python argument.
source_image_folder = "CelebAMask-HQ"

# This folder contains a copy of data from source_image_folder defined just above,
# but in a structure permitting to train and test easily.
# All good and bad folders will be created inside this folder. 
# We can safely delete this folder if we want to run again the script,
# by example when we change the criteria to classify Bad and Good data.
processed_image_folder = "Data"

# You need to use os.path.join function in order to concatenate File or Folder PATH, 
# because this latter is OS dependent. They are not the same in Windows and Linux.
goodTest_destination = os.path.join(processed_image_folder, "Testing", "Good")
badTest_destination = os.path.join(processed_image_folder, "Testing", "Bad")
good_destination = os.path.join(processed_image_folder, "Training", "Good")
bad_destination = os.path.join(processed_image_folder, "Training", "Bad")

# Prepare the directory to store Train/Test, Good/Bad data.
# If the folder already exist, the function will throw exception, so that you will
# not be able to accidently run this script twice. If you really want to run it 
# again, you have to delete the whole processed_image_folder
os.makedirs(goodTest_destination)
os.makedirs(badTest_destination)
os.makedirs(good_destination)
os.makedirs(bad_destination)

with open(os.path.join(source_image_folder, "CelebAMask-HQ-pose-anno.txt")) as p:
    for line in p:
        data = line.strip().split('.jpg ', 1)
        try:
            yaw, pitch, roll = (item.strip() for item in data[1].split(' ', 2))
            yaw = float(yaw)
            pitch = float(pitch)
            roll = float(roll)
            if yaw > -15 and yaw < 15 and pitch > -20 and pitch < 5 and roll > -5 and roll < 5:
                good[data[0]] = {"data": [yaw, pitch, roll, -1], "status": "1"}
            else:
                bad[data[0]] = {"data": [yaw, pitch, roll, -1], "status": "0"}
        except IndexError:
            continue

with open(os.path.join(source_image_folder, "CelebAMask-HQ-attribute-anno.txt")) as p:
    for line in p:
        data = line.strip().split('.jpg  ', 1)
        try:
            Five_Clock_Shadow, Arched_Eyebrows, Attractive, Bags_Under_Eyes, Bald, Bangs, Big_Lips, Big_Nose, Black_Hair, Blond_Hair, Blurry, Brown_Hair, Bushy_Eyebrows, Chubby, Double_Chin, Eyeglasses, Goatee, Gray_Hair, Heavy_Makeup, High_Cheekbones, Male, Mouth_Slightly_Open, Mustache, Narrow_Eyes, No_Beard, Oval_Face, Pale_Skin, Pointy_Nose, Receding_Hairline, Rosy_Cheeks, Sideburns, Smiling, Straight_Hair, Wavy_Hair, Wearing_Earrings, Wearing_Hat, Wearing_Lipstick, Wearing_Necklace, Wearing_Necktie, Young = (item.strip() for item in data[1].split(' ', 39))
            if (int(Eyeglasses) == 1 or int(Blurry) == 1 or int(Wearing_Hat) == 1) and data[0] in good:
                bad[data[0]] = good[data[0]]
                bad[data[0]]["status"] = "0"
                bad[data[0]]["data"][-1] = 1
                del (good[data[0]])
        except IndexError:
            continue

for i in good.keys():
    dest = shutil.copy(os.path.join(source_image_folder, "CelebA-HQ-img", '{}.jpg'.format(i)), good_destination)
    print("File copied: ", dest)

for i in bad.keys():
    dest = shutil.copy(os.path.join(source_image_folder, "CelebA-HQ-img", '{}.jpg'.format(i)), bad_destination)
    print("File copied: ", dest)

# Hand detection
Img_size = 50
mpHands = mp.solutions.hands
hands = mpHands.Hands()
for img in os.listdir(good_destination):
    try:
        img_array = cv2.imread(os.path.join(good_destination,img))
        imgRGB = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        hand_check = hands.process(imgRGB)
        if hand_check.multi_hand_landmarks:
            print("Removing hands", img)
            dest = shutil.move(os.path.join(good_destination,img), bad_destination)
    except Exception as e:
        pass

# Create good test file
count = 0
for i in os.listdir(good_destination):
    if count < 3000:
        dest = shutil.move(os.path.join(good_destination,i), goodTest_destination)
        print("File moved to Test folder", dest)
        count+=1
    else:
        break

# Create bad test file
count = 0
for i in os.listdir(bad_destination):
    if count < 3000:
        dest = shutil.move(os.path.join(bad_destination,i), badTest_destination)
        print("File moved to Test folder", dest)
        count+=1
    else:
        break

