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
from lung_dataset import Lung_Dataset
from torch.utils.data import Dataset, DataLoader

import model_binary
import model_three_class



# train_ld = Lung_Dataset('train', 'three_class', augmentation='hist_equal')
# from aug_data_generator import plot_comparison
# plot_comparison('hist_equal')

# ld_train1 = Lung_Dataset("train", "binary1")
# ld_val1 = Lung_Dataset("val", "binary1")
# ld_test1 = Lung_Dataset("test", "binary1")

# ld_train2 = Lung_Dataset("train", "binary2")
# ld_val2 = Lung_Dataset("val", "binary2")
# ld_test2 = Lung_Dataset("test", "binary2")
# print(ld_train1.describe())


#====================================================================================
#=====================         CASCADED BINARY MODEL         ========================
#====================================================================================

# batch_size = 128
# n_epochs = 20
# lr = 0.00001
# saved_model_path =  "./model/"
# device = "cpu"
# # device = "cuda"

# classifier1 = model_binary.Model_binary_bn()
# classifier2 = model_binary.Model_binary_bn()

# classifier1.to(device)
# classifier2.to(device)

# train_loader1 = DataLoader(ld_train1, batch_size=batch_size, shuffle=True)
# val_loader1 = DataLoader(ld_val1, batch_size=batch_size, shuffle=True)
# test_loader1 = DataLoader(ld_test1, batch_size=batch_size, shuffle=True)

# train_loader2 = DataLoader(ld_train2, batch_size=batch_size, shuffle=True)
# val_loader2 = DataLoader(ld_val2, batch_size=batch_size, shuffle=True)
# test_loader2 = DataLoader(ld_test2, batch_size=batch_size, shuffle=True)


# trained_classifier1 = model_binary.train(classifier1, 'bn_class_1', batch_size, n_epochs, lr, train_loader1, val_loader1, saved_model_path, device)
# model_binary.test_model(trained_classifier1,test_loader1, device)
# trained_classifier2 = model_binary.train(classifier2, 'bn_class_2', batch_size, n_epochs, lr, train_loader2, val_loader2, saved_model_path, device)


#====================================================================================
#=======================         VALIDATION DISPLAY         =========================
#====================================================================================
model = model_three_class.ModelThreeClassLite()


#====================================================================================
#=======================         VALIDATION DISPLAY         =========================
#====================================================================================
ld_val = Lung_Dataset("val", "three_class")
val_dataloader = DataLoader(ld_val)

fig, axes = plt.subplots(8, 3,figsize=(10,10))

labels = ["normal", "infected covid", "infected non-covid"]
for i, (img, label) in enumerate(val_dataloader):
    groundTruth = labels[label]
    output = model(img)
    prediction = labels[output.max(dim=1)[1]]
    # print(prediction)

    axes[i % 8, label].imshow(img.reshape(150, 150))
    axes[i % 8, label].set_title(f"Ground Truth: {groundTruth} \n Prediction: {prediction}", size=7)

# plt.subplots_adjust(hspace=20)
plt.tight_layout()
plt.show()
