from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms


class ModelThreeClass(nn.Module):
    def __init__(self):
        super(ModelThreeClass, self).__init__()

        inp_img_size = 150
        kernel_size = 7
        stride = 1
        padding = 0
        conv1_output_channels = 16
        conv2_output_channels = 16
        conv3_output_channels = 16

        # conv1_output_size = (inp_img_size-kernel_size+2*padding)/stride + 1;
        # input img: 150x150
        self.conv1 = nn.Conv2d(1, conv1_output_channels, kernel_size, stride)
        # input: 144x144x4
        self.conv2 = nn.Conv2d(conv1_output_channels,
                               conv2_output_channels, kernel_size, stride, padding=3)
        # input: 144x144x8
        self.pool = nn.MaxPool2d(2, 2)
        # input: 72x72x8
        self.conv3 = nn.Conv2d(conv2_output_channels,
                               conv3_output_channels, kernel_size, stride, padding=3)
        # input: 72x72x16
        self.fc1 = nn.Linear(20736, 1024)
        self.fc2 = nn.Linear(1024, 120)
        self.fc3 = nn.Linear(120, 3)
        # self.out = nn.Softmax(3)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        residual = x
        x = F.relu(self.conv2(x))
        x += residual

        x = self.pool(x)
        x = F.relu(self.conv3(x))

        x = self.pool(x)
        x = torch.flatten(x, 1)        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        x = self.fc3(x)
        return F.softmax(x, dim=1)
        # output = F.log_softmax(x, dim=1)
        # return torch.exp(output)
        # return output


device = torch.device('cuda:0')

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

train_ld = Lung_Dataset('train', 'three_class')

# train_ld.describe()

train_loader = DataLoader(train_ld, batch_size=512, shuffle=True)

lossGraph = []
def train_model(model, epochs=50):
    model.train()

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    # maxCoutn = 10
    # curCount = 0
    
    for i in range(epochs):
        running_loss = 0
        for batch_idx, (images_data, target_labels) in enumerate(train_loader):
            images_data = images_data.to(device)
            target_labels = target_labels.to(device)
            
            optim.zero_grad()

            prediction = model(images_data)
            loss = nn.CrossEntropyLoss()
            output = loss(prediction, target_labels)
            output.backward()
            
            optim.step()  # gradient descent

            # print('pred:', np.argmax(prediction.detach().numpy(), axis=1))
            # print('target:', target_labels)
            # print(output.item())
            running_loss += output.item()

            # curCount += 1
            # if curCount > maxCoutn:
            #     break

        lossGraph.append(running_loss)
        print(f'Epoch: {i} Loss: {running_loss}')    


model_three_class = ModelThreeClass()
model_three_class.to(device)
train_model(model_three_class)
# print(torch.cuda.is_available())