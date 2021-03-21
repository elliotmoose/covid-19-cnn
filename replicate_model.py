# Matplotlib
import matplotlib.pyplot as plt
import seaborn as sn
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
from lung_dataset import Lung_Dataset
from torch.utils.data import Dataset, DataLoader

import model_binary
import model_three_class
import model_trainer
import metrics


ld_test3 = Lung_Dataset("test", "three_class")
test_loader3 = DataLoader(ld_test3, batch_size=1, shuffle=True)

classifier1 = torch.load("./model/bn_class_1-2021-03-21 15_29_22.328705-b128-e19.pt")
classifier2 = torch.load("./model/bn_class_2-2021-03-21 15_31_01.588591-b128-e29.pt")

accuracy, confusion_matrix = model_trainer.test_binary(classifier1, classifier2, test_loader3, device)
#show confusion matrix
metrics.show_confusion(confusion_matrix)