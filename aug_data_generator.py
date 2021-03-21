import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch

import os

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
        #INSERT FLIP IMPLEMTATION HERE
        pass

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
class AugmentationFns():
    @staticmethod
    def histEqualise(img):
        """
        img: cv for img, as returned by cv.imread
        """
        src = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return cv.equalizeHist(src)
