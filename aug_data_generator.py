import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch

import os
from skimage import io
from skimage.transform import rotate, AffineTransform, warp
import random
import glob

def create_dir(dir):
    os.mkdir(dir)

def read_img(img_path):
    return cv.imread(img_path) #read as ndarray with nd = 2

def create_img(path, img):
    cv.imwrite(path, img)

def images_in_dir(dir):
    return os.listdir(dir)
    
def generate_augmented_data(augmentation):
    if augmentation == "HIST_EQUALISE":
        augment_name = 'hist_equal'
        augment_fn = AugmentationFns.histEqualise

        create_dir(f'dataset/{augment_name}')
        for group_val in ['train', 'val', 'test']:
            create_dir(f'dataset/{augment_name}/{group_val}')
            for class_val in ['infected', 'normal']:
                create_dir(f'dataset/{augment_name}/{group_val}/{class_val}')
                if class_val == 'infected':
                    for covid_class_val in ['covid', 'non-covid']:
                        create_dir(f'dataset/{augment_name}/{group_val}/{class_val}/{covid_class_val}')
                        for i, img_name in enumerate(images_in_dir(f'dataset/vanilla/{group_val}/{class_val}/{covid_class_val}')):
                            img_path = f'dataset/vanilla/{group_val}/{class_val}/{covid_class_val}/{img_name}'
                            img = read_img(img_path)
                            augmented_img = augment_fn(img)
                            create_img(f'dataset/{augment_name}/{group_val}/{class_val}/{covid_class_val}/{img_name}', augmented_img)
                else:
                    for i, img_name in enumerate(images_in_dir(f'dataset/vanilla/{group_val}/{class_val}')):
                        img_path = f'dataset/vanilla/{group_val}/{class_val}/{img_name}'
                        img = read_img(img_path)
                        augmented_img = augment_fn(img)
                        create_img(f'dataset/{augment_name}/{group_val}/{class_val}/{img_name}', augmented_img)
    elif augmentation == "FLIP":
        count = 0
        for settype in os.listdir("./dataset/vanilla/"):
            print(settype)
            if settype == 'train' or settype == 'val' or settype == 'test':
                folder = "./dataset/vanilla/" + settype + "/normal/"
                newfolder = "./dataset/flip/" + settype + "/normal/"
                if not os.path.exists(newfolder):
                    os.makedirs(newfolder)
                    print(newfolder)
                    count = 0
                    for image in os.listdir(folder):
                        imagepath = folder + "/" + image
                        img = cv.imread(imagepath)
                        flip = cv.flip(img, 1)
                        cv.imwrite("./dataset/flip/" + settype +
                                    "/normal/" + str(count) + ".jpg", flip)
                        count += 1

                folder = "./dataset/vanilla/" + settype + "/infected/covid/"
                newfolder = "./dataset/flip/" + settype + "/infected/covid/"
                if not os.path.exists(newfolder):
                    os.makedirs(newfolder)
                    print(newfolder)
                    count = 0
                    for image in os.listdir(folder):
                        imagepath = folder + "/" + image
                        img = cv.imread(imagepath)
                        flip = cv.flip(img, 1)
                        cv.imwrite("./dataset/flip/" + settype +
                                    "/infected/covid/" + str(count) + ".jpg", flip)
                        count += 1

                folder = "./dataset/vanilla/" + settype + "/infected/non-covid/"
                newfolder = "./dataset/flip/" + settype + "/infected/non-covid/"
                if not os.path.exists(newfolder):
                    os.makedirs(newfolder)
                    print(newfolder)
                    count = 0
                    for image in os.listdir(folder):
                        imagepath = folder + "/" + image
                        img = cv.imread(imagepath)
                        flip = cv.flip(img, 1)
                        cv.imwrite("./dataset/flip/" + settype +
                                    "/infected/non-covid/" + str(count) + ".jpg", flip)
                        count += 1
    elif augmentation == "BRIGHTNESS":
        count = 0
        for settype in os.listdir("./dataset/vanilla/"):
            print(settype)
            if settype == 'train' or settype == 'val' or settype == 'test':
                folder = "./dataset/vanilla/" + settype + "/normal/"
                newfolder = "./dataset/brightness/" + settype + "/normal/"
                if not os.path.exists(newfolder):
                    os.makedirs(newfolder)
                    print(newfolder)
                    count = 0
                    for image in os.listdir(folder):
                        imagepath = folder + "/" + image
                        img = cv.imread(imagepath)
                        brightness = random.uniform(0.1,2)
                        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
                        hsv = np.array(hsv, dtype=np.float64)
                        hsv[:,:,1] = hsv[:,:,1]*brightness
                        hsv[:,:,1][hsv[:,:,1]>255] = 255
                        hsv[:,:,2] = hsv[:,:,2]*brightness
                        hsv[:,:,2][hsv[:,:,2]>255] = 255
                        hsv = np.array(hsv, dtype=np.uint8)
                        adjusted = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
                        cv.imwrite("./dataset/brightness/" + settype +
                                    "/normal/" + str(count) + ".jpg", adjusted)
                        count += 1

                folder = "./dataset/vanilla/" + settype + "/infected/covid/"
                newfolder = "./dataset/brightness/" + settype + "/infected/covid/"
                if not os.path.exists(newfolder):
                    os.makedirs(newfolder)
                    print(newfolder)
                    count = 0
                    for image in os.listdir(folder):
                        imagepath = folder + "/" + image
                        img = cv.imread(imagepath)
                        brightness = random.uniform(0.1,2)
                        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
                        hsv = np.array(hsv, dtype=np.float64)
                        hsv[:,:,1] = hsv[:,:,1]*brightness
                        hsv[:,:,1][hsv[:,:,1]>255] = 255
                        hsv[:,:,2] = hsv[:,:,2]*brightness
                        hsv[:,:,2][hsv[:,:,2]>255] = 255
                        hsv = np.array(hsv, dtype=np.uint8)
                        adjusted = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
                        cv.imwrite("./dataset/brightness/" + settype +
                                    "/infected/covid/" + str(count) + ".jpg", adjusted)
                        count += 1

                folder = "./dataset/vanilla/" + settype + "/infected/non-covid/"
                newfolder = "./dataset/brightness/" + settype + "/infected/non-covid/"
                if not os.path.exists(newfolder):
                    os.makedirs(newfolder)
                    print(newfolder)
                    count = 0
                    for image in os.listdir(folder):
                        imagepath = folder + "/" + image
                        img = cv.imread(imagepath)
                        brightness = random.uniform(0.1,2)
                        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
                        hsv = np.array(hsv, dtype=np.float64)
                        hsv[:,:,1] = hsv[:,:,1]*brightness
                        hsv[:,:,1][hsv[:,:,1]>255] = 255
                        hsv[:,:,2] = hsv[:,:,2]*brightness
                        hsv[:,:,2][hsv[:,:,2]>255] = 255
                        hsv = np.array(hsv, dtype=np.uint8)
                        adjusted = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
                        cv.imwrite("./dataset/brightness/" + settype +
                                    "/infected/non-covid/" + str(count) + ".jpg", adjusted)
                        count += 1


def plot_comparison(augmentation):
    num = 5
    fig, axes = plt.subplots(num, 2, figsize=(10,10))

    for i in range(num):
        img_a = read_img(f'dataset/vanilla/train/infected/covid/{i}.jpg')
        img_b = read_img(f'dataset/{augmentation}/train/infected/covid/{i}.jpg')
        axes[i, 0].imshow(img_a)
        axes[i, 0].title.set_text(f'{i}.jpg')
        axes[i, 1].imshow(img_b)
        axes[i, 1].title.set_text(f'{i}.jpg')
    plt.tight_layout()
    plt.show()

class Augmentations():
    HIST_EQUALISE="HIST_EQUALISE"
    FLIP="FLIP"
    BRIGHTNESS="BRIGHTNESS"
class AugmentationFns():
    @staticmethod
    def histEqualise(img):
        """
        img: cv for img, as returned by cv.imread
        """
        src = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return cv.equalizeHist(src)
