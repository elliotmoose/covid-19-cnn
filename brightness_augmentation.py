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
        newfolder = "./dataset/brightness/" + settype + "/normal/"
        if not os.path.exists(newfolder):
            os.makedirs(newfolder)
            print(newfolder)
            count = 0
            for image in os.listdir(folder):
                imagepath = folder + "/" + image
                img = cv2.imread(imagepath)
                brightness = random.uniform(0.1,2)
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                hsv = np.array(hsv, dtype=np.float64)
                hsv[:,:,1] = hsv[:,:,1]*brightness
                hsv[:,:,1][hsv[:,:,1]>255] = 255
                hsv[:,:,2] = hsv[:,:,2]*brightness
                hsv[:,:,2][hsv[:,:,2]>255] = 255
                hsv = np.array(hsv, dtype=np.uint8)
                adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                cv2.imwrite("./dataset/brightness/" + settype +
                            "/normal/" + str(count) + ".jpg", adjusted)
                count += 1

        folder = "./dataset/" + settype + "/infected/covid/"
        newfolder = "./dataset/brightness/" + settype + "/infected/covid/"
        if not os.path.exists(newfolder):
            os.makedirs(newfolder)
            print(newfolder)
            count = 0
            for image in os.listdir(folder):
                imagepath = folder + "/" + image
                img = cv2.imread(imagepath)
                brightness = random.uniform(0.1,2)
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                hsv = np.array(hsv, dtype=np.float64)
                hsv[:,:,1] = hsv[:,:,1]*brightness
                hsv[:,:,1][hsv[:,:,1]>255] = 255
                hsv[:,:,2] = hsv[:,:,2]*brightness
                hsv[:,:,2][hsv[:,:,2]>255] = 255
                hsv = np.array(hsv, dtype=np.uint8)
                adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                cv2.imwrite("./dataset/brightness/" + settype +
                            "/infected/covid/" + str(count) + ".jpg", adjusted)
                count += 1

        folder = "./dataset/" + settype + "/infected/non-covid/"
        newfolder = "./dataset/brightness/" + settype + "/infected/non-covid/"
        if not os.path.exists(newfolder):
            os.makedirs(newfolder)
            print(newfolder)
            count = 0
            for image in os.listdir(folder):
                imagepath = folder + "/" + image
                img = cv2.imread(imagepath)
                brightness = random.uniform(0.1,2)
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                hsv = np.array(hsv, dtype=np.float64)
                hsv[:,:,1] = hsv[:,:,1]*brightness
                hsv[:,:,1][hsv[:,:,1]>255] = 255
                hsv[:,:,2] = hsv[:,:,2]*brightness
                hsv[:,:,2][hsv[:,:,2]>255] = 255
                hsv = np.array(hsv, dtype=np.uint8)
                adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                cv2.imwrite("./dataset/brightness/" + settype +
                            "/infected/non-covid/" + str(count) + ".jpg", adjusted)
                count += 1

        