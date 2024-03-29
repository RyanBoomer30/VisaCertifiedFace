import cv2
import tensorflow as tf
import os
import numpy as np
from sklearn.metrics import roc_curve, RocCurveDisplay, roc_auc_score
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

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

result = model.evaluate(x_test, y_test, batch_size=32)

y_pred = model.predict(x_test)

fpr, tpr, threshold = roc_curve(y_test, y_pred)

fnr = 1 - tpr
eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
print(EER)

# roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
# plt.show()