import cv2
import numpy as np
from skimage import io
from skimage.transform import rotate, AffineTransform, warp
import matplotlib.pyplot as plt
import random
import cv2
import os
import glob

count = 0
for settype in os.listdir("./dataset/"):
    print(settype)
    if settype == 'train' or settype == 'val' or settype == 'test':
        folder = "./dataset/" + settype + "/normal/"
        newfolder = "./dataset/flipped/" + settype + "/normal/"
        if not os.path.exists(newfolder):
            os.makedirs(newfolder)
            print(newfolder)
            count = 0
            for image in os.listdir(folder):
                imagepath = folder + "/" + image
                img = cv2.imread(imagepath)
                flipped = cv2.flip(img, 1)
                cv2.imwrite("./dataset/flipped/" + settype +
                            "/normal/" + str(count) + ".jpg", flipped)
                count += 1

        folder = "./dataset/" + settype + "/infected/covid/"
        newfolder = "./dataset/flipped/" + settype + "/infected/covid/"
        if not os.path.exists(newfolder):
            os.makedirs(newfolder)
            print(newfolder)
            count = 0
            for image in os.listdir(folder):
                imagepath = folder + "/" + image
                img = cv2.imread(imagepath)
                flipped = cv2.flip(img, 1)
                cv2.imwrite("./dataset/flipped/" + settype +
                            "/infected/covid/" + str(count) + ".jpg", flipped)
                count += 1

        folder = "./dataset/" + settype + "/infected/non-covid/"
        newfolder = "./dataset/flipped/" + settype + "/infected/non-covid/"
        if not os.path.exists(newfolder):
            os.makedirs(newfolder)
            print(newfolder)
            count = 0
            for image in os.listdir(folder):
                imagepath = folder + "/" + image
                img = cv2.imread(imagepath)
                flipped = cv2.flip(img, 1)
                cv2.imwrite("./dataset/flipped/" + settype +
                            "/infected/non-covid/" + str(count) + ".jpg", flipped)
                count += 1
