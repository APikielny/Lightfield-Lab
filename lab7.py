import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import skimage
from skimage import transform as tf
import math


def epi(row):
    output = np.zeros((473, 1255, 3))
    fileNames = os.listdir('./train_100to200')
    fileNames.sort()
    # print(fileNames)
    for (index, img_path) in enumerate(fileNames):
        # print(img_path)
        img = cv2.imread('./train_100to200/' + img_path).astype(np.float32)
        # print(img)
        img_slice = img[row]
        output[index] = img_slice
    return output #should be for one row of input images

def epiAverage(epiImage):
    return np.mean(epiImage, axis = 0)

def epiStacked():
    output = np.zeros((101, 473, 1255, 3))
    fileNames = os.listdir('./train_100to200')
    fileNames.sort()
    # print(fileNames)
    for (index, img_path) in enumerate(fileNames):
        # print(img_path)
        img = cv2.imread('./train_100to200/' + img_path).astype(np.float32)
        # print(img)
        # transformation = tf.SimilarityTransform(scale=1, rotation=math.pi/2,
        #                        translation=(0, 1))
        # warped = skimage.transform.warp(img, transformation)
        output[index] = img

        # output[index] = img
    return output #should be for one row of input images

def epiAverageStacked(epiImage):
    return np.mean(epiImage, axis = 0)

####
# test one epi row

# testEpi = epi(5)
# avg = epiAverage(testEpi)
# print(testEpi)
# plt.imshow(avg / 255)
# plt.show()

####
# visualize epi lines sequentially
# ROWS = 473
# output = np.zeros((ROWS, 1255, 3))
# for row in range(ROWS):
#     epiRow = epi(row)
#     cv2.imshow("epi row", epiRow / 255)
#     cv2.waitKey(delay = 1)

### 
## average (squash epi rows)
epiStackedImg = epiStacked()
# print("stacked img shape", epiStackedImg.shape)
avg = epiAverageStacked(epiStackedImg)
# print("avg shape", avg.shape)
# print(avg)
cv2.imshow("squashed output", avg / 255)
cv2.waitKey(0)

## TODO use skimage transform to transform epi rows, to change focus of the image