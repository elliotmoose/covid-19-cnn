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
train_ld = Lung_Dataset('train', 'three_class', augmentation='vanilla')
# from aug_data_generator import plot_comparison
# plot_comparison('hist_equal')



#====================================================================================
#=====================         CASCADED BINARY MODEL         ========================
#====================================================================================

ld_train1 = Lung_Dataset("train", "binary1")
ld_val1 = Lung_Dataset("val", "binary1")
ld_test1 = Lung_Dataset("test", "binary1")

ld_train2 = Lung_Dataset("train", "binary2")
ld_val2 = Lung_Dataset("val", "binary2")
ld_test2 = Lung_Dataset("test", "binary2")
print(ld_train1.describe())

batch_size = 128
n_epochs = 20
lr = 0.00001
saved_model_path =  "./model/"
# device = "cpu"
device = "cuda"

classifier1 = model_binary.Model_binary_bn(classifier=1)
classifier2 = model_binary.Model_binary_bn(classifier=2)

classifier1.to(device)
classifier2.to(device)

train_loader1 = DataLoader(ld_train1, batch_size=batch_size, shuffle=True)
val_loader1 = DataLoader(ld_val1, batch_size=batch_size, shuffle=True)
test_loader1 = DataLoader(ld_test1, batch_size=batch_size, shuffle=True)

train_loader2 = DataLoader(ld_train2, batch_size=batch_size, shuffle=True)
val_loader2 = DataLoader(ld_val2, batch_size=batch_size, shuffle=True)
test_loader2 = DataLoader(ld_test2, batch_size=batch_size, shuffle=True)


classifier1_latest_weights, classifier1_accuracy_weights, classifier1_recall_weights = model_trainer.train(classifier1, 'bn_class_1', batch_size, n_epochs, lr, train_loader1, test_loader1, saved_model_path, device)
classifier2_latest_weights, classifier2_accuracy_weights, classifier2_recall_weights  = model_trainer.train(classifier2, 'bn_class_2', batch_size, 30, lr, train_loader2, test_loader2, saved_model_path, device)


ld_test3 = Lung_Dataset("test", "three_class")
test_loader3 = DataLoader(ld_test3, batch_size=1, shuffle=True)

classifier1.load_state_dict(classifier1_accuracy_weights)
classifier2.load_state_dict(classifier2_accuracy_weights)
accuracy, confusion_matrix = model_trainer.test_binary(classifier1, classifier2, test_loader3, device)
#show confusion matrix
metrics.show_confusion(confusion_matrix)

classifier1.load_state_dict(classifier1_recall_weights)
classifier2.load_state_dict(classifier2_recall_weights)
accuracy, confusion_matrix = model_trainer.test_binary(classifier1, classifier2, test_loader3, device)
#show confusion matrix
metrics.show_confusion(confusion_matrix)

#====================================================================================
#========================         THREE CLASSIFIER         ==========================
#====================================================================================
batch_size = 512
n_epochs = 25
lr = 1e-3
ld_train3 = Lung_Dataset("train", "three_class")
ld_val3 = Lung_Dataset("val", "three_class")
ld_test3 = Lung_Dataset("test", "three_class")

train_loader3 = DataLoader(ld_train3, batch_size=batch_size, shuffle=True)
val_loader3 = DataLoader(ld_val3, batch_size=batch_size, shuffle=True)
test_loader3 = DataLoader(ld_test3, batch_size=batch_size, shuffle=True)

three_classifier = model_three_class.ModelThreeClass()
three_classifier.to(device)

three_class_latest_weights, three_class_accuracy_weights, three_class_recall_weights = model_trainer.train(three_classifier, 'three_class', batch_size, n_epochs, lr, train_loader3, test_loader3, saved_model_path, device, num_classes=3)

three_classifier.load_state_dict(three_class_accuracy_weights)
accuracy, confusion_matrix = model_trainer.test(three_classifier,test_loader3, device, num_classes=3)
#show confusion matrix
metrics.show_confusion(confusion_matrix)

three_classifier.load_state_dict(three_class_recall_weights)
accuracy, confusion_matrix = model_trainer.test(three_classifier,test_loader3, device, num_classes=3)
#show confusion matrix
metrics.show_confusion(confusion_matrix)

#====================================================================================
#=======================         VALIDATION DISPLAY         =========================
#====================================================================================
# ld_val = Lung_Dataset("val", "three_class")
# val_dataloader = DataLoader(ld_val)

# fig, axes = plt.subplots(8, 3,figsize=(10,10))

# labels = ["normal", "infected covid", "infected non-covid"]
# trained_three_classifier.eval()
# for i, (img, label) in enumerate(val_dataloader):
#     img, label = img.to(device), label.to(device)
#     groundTruth = labels[label]
#     output = trained_three_classifier(img)
#     prediction = labels[output.max(dim=1)[1]]
#     # print(prediction)

#     ax = axes[i % 8, label]
#     ax.imshow(img.cpu().reshape(150, 150))
#     ax.set_title(f"Ground Truth: {groundTruth} \n Prediction: {prediction}", size=7)
#     ax.xaxis.set_visible(False)
#     ax.yaxis.set_visible(False)

# # plt.subplots_adjust(hspace=20)
# plt.tight_layout()
# plt.show()
