# Matplotlib
import matplotlib.pyplot as plt
# Numpy
import numpy as np
# Pillow
from PIL import Image
# Torch
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import os


class Lung_Dataset(Dataset):
    """
    Generic Dataset class.
    """

    def __init__(self, group, model_type):
        """
        Constructor for generic Dataset class - simply assembles
        the important parameters in attributes.
        """
        self.model_type = model_type

        # All images are of size 150 x 150
        self.img_size = (150, 150)

        # Only two classes will be considered here (normal and infected)

        # if self.model_type == "three_class":
        # elif self.model_type == "binary":
        #     self.classes = {0: 'normal', 1: 'infected', 2:'infected_covid', 3: 'infected_noncovid'}
        self.classes = {0: 'normal',
                        1: 'infected_covid', 2: 'infected_noncovid'}

        # The dataset has been split in training, testing and validation datasets
        self.groups = group
        # self.groups = ['train', 'test', 'val']

        # Number of images in each part of the dataset
        train_normal_no = len(os.listdir('./dataset/train/normal'))
        train_infected_covid_no = len(
            os.listdir('./dataset/train/infected/covid'))
        train_infected_noncovid_no = len(
            os.listdir('./dataset/train/infected/non-covid'))

        val_normal_no = len(os.listdir('./dataset/val/normal'))
        val_infected_covid_no = len(os.listdir('./dataset/val/infected/covid'))
        val_infected_noncovid_no = len(
            os.listdir('./dataset/val/infected/non-covid'))

        test_normal_no = len(os.listdir('./dataset/test/normal'))
        test_infected_covid_no = len(
            os.listdir('./dataset/test/infected/covid'))
        test_infected_noncovid_no = len(
            os.listdir('./dataset/test/infected/non-covid'))

        self.dataset_numbers = {'train_normal': train_normal_no,
                                'train_infected_covid': train_infected_covid_no,
                                'train_infected_noncovid': train_infected_noncovid_no,
                                'val_normal': val_normal_no,
                                'val_infected_covid': val_infected_covid_no,
                                'val_infected_noncovid': val_infected_noncovid_no,
                                'test_normal': test_normal_no,
                                'test_infected_covid': test_infected_covid_no,
                                'test_infected_noncovid': test_infected_noncovid_no}

        # Path to images for different parts of the dataset
        self.dataset_paths = {'train_normal': './dataset/train/normal',
                              'train_infected_covid': './dataset/train/infected/covid',
                              'train_infected_noncovid': './dataset/train/infected/non-covid',
                              'val_normal': './dataset/val/normal',
                              'val_infected_covid': './dataset/val/infected/covid',
                              'val_infected_noncovid': './dataset/val/infected/non-covid',
                              'test_normal': './dataset/test/normal',
                              'test_infected_covid': './dataset/test/infected/covid',
                              'test_infected_noncovid': './dataset/test/infected/non-covid'}

    def describe(self):
        """
        Descriptor function.
        Will print details about the dataset when called.
        """

        # Generate description
        msg = "This is the Lung Dataset used for the Small Project Demo in the 50.039 Deep Learning class"
        msg += " in Feb-March 2021. \n"
        msg += "It contains a total of {} images, ".format(
            sum(self.dataset_numbers.values()))
        msg += "of size {} by {}.\n".format(self.img_size[0], self.img_size[1])
        msg += "Images have been split in three groups: training, testing and validation sets.\n"
        msg += "The images are stored in the following locations "
        msg += "and each one contains the following number of images:\n"
        for key, val in self.dataset_paths.items():
            msg += " - {}, in folder {}: {} images.\n".format(
                key, val, self.dataset_numbers[key])
        print(msg)

    def open_img(self, group_val, class_val, index_val):
        """
        Opens image with specified parameters.

        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal' or 'infected'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.

        Returns loaded image as a normalized Numpy array.
        """
        # Asserts checking for consistency in passed parameters
        err_msg = "Error - group_val variable should be set to 'train', 'test' or 'val'."
        assert group_val in self.groups, err_msg

        err_msg = "Error - class_val variable should be set to 'normal' or 'infected'."
        assert class_val in self.classes.values(), err_msg

        max_val = self.dataset_numbers['{}_{}'.format(group_val, class_val)]
        err_msg = "Error - index_val variable should be an integer between 0 and the maximal number of images."
        err_msg += f"\n(In {group_val}/{class_val}, you have {max_val} images. Tried to retrieve index {index_val})"
        assert isinstance(index_val, int), err_msg
        assert index_val >= 0 and index_val <= max_val, err_msg

        # Open file as before
        path_to_file = '{}/{}.jpg'.format(
            self.dataset_paths['{}_{}'.format(group_val, class_val)], index_val)

        with open(path_to_file, 'rb') as f:
            # Convert to Numpy array and normalize pixel values by dividing by 255.
            im = np.asarray(Image.open(f))/255
        f.close()
        return im

    def show_img(self, group_val, class_val, index_val):
        """
        Opens, then displays image with specified parameters.

        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal' or 'infected'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        """

        # Open image
        im = self.open_img(group_val, class_val, index_val)

        # Display
        plt.imshow(im)

    def __len__(self):
        """
        Length special method, returns the number of images in dataset.
        """

        # Length function
        keys = self.dataset_numbers.keys()

        sum = 0
        for key in keys:
            if key[:len(self.groups)] == self.groups:
                sum += self.dataset_numbers[key]
        return sum

    def __getitem__(self, index):
        """
        Getitem special method.

        Expects an integer value index, between 0 and len(self) - 1.

        Returns the image and its label as a one hot vector, both
        in torch tensor format in dataset.
        """
        # Get item special method for binary1 model
        first_val = self.dataset_numbers[f'{self.groups}_normal']
        second_val = first_val + \
            self.dataset_numbers[f'{self.groups}_infected_covid']

        if self.model_type == "binary":

            if index < first_val:
                class_val = 'normal'
                # label = torch.Tensor([1, 0])
                label = 0

            elif index < second_val:
                class_val = 'infected_covid'
                index = index - first_val
                # label = torch.Tensor([0, 1])
                label = 1

            else:
                class_val = 'infected_noncovid'
                index = index - second_val
                # label = torch.Tensor([0, 1])
                label = 1

        elif self.model_type == "three_class":
            # Get item special method

            if index < first_val:
                class_val = 'normal'
                label = 0
                # label = torch.Tensor([0])

            elif index < second_val:
                class_val = 'infected_covid'
                index = index - first_val
                label = 1
                # label = torch.Tensor([1])

            else:
                class_val = 'infected_noncovid'
                index = index - second_val
                label = 2
                # label = torch.Tensor([2])

        im = self.open_img(self.groups, class_val, index)
        im = transforms.functional.to_tensor(np.array(im)).float()
        return im, label
