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

from model_three_class import ModelThreeClass

from lung_dataset import Lung_Dataset
from torch.utils.data import Dataset, DataLoader

train_ld = Lung_Dataset('train', 'three_class')

# train_ld.describe()

train_loader = DataLoader(train_ld, batch_size=512, shuffle=True)


def train_model(model, epochs=10):
    model.train()

    # maxCoutn = 10
    # curCount = 0
    for i in range(epochs):
        running_loss = 0
        for batch_idx, (images_data, target_labels) in enumerate(train_loader):
            prediction = model(images_data)
            loss = nn.CrossEntropyLoss()
            output = loss(prediction, target_labels)
            output.backward()
            optim = torch.optim.Adam(model.parameters(), lr=1e-3)
            optim.step()  # gradient descent

            # print('pred:', np.argmax(prediction.detach().numpy(), axis=1))
            # print('target:', target_labels)
            print(output.item())
            running_loss += output.item()

            # curCount += 1
            # if curCount > maxCoutn:
            #     break

        print(f'Epoch: {i} Loss: {running_loss}')


model_three_class = ModelThreeClass()
train_model(model_three_class)


# print(torch.cuda.is_available())
