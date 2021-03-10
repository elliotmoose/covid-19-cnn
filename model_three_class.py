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
        kernel_size = 3
        stride = 1
        padding = 0
        conv1_output_channels = 4
        conv2_output_channels = 8
        conv3_output_channels = 16

        # conv1_output_size = (inp_img_size-kernel_size+2*padding)/stride + 1;
        # input img: 150x150
        self.conv1 = nn.Conv2d(1, conv1_output_channels, kernel_size, stride)
        # input: 148x148x4
        self.conv2 = nn.Conv2d(conv1_output_channels,
                               conv2_output_channels, kernel_size, stride)
        # input: 146x146x8
        self.pool = nn.MaxPool2d(2, 2)
        # input: 73x73x8
        self.conv3 = nn.Conv2d(conv2_output_channels,
                               conv3_output_channels, kernel_size, stride)
        # input: 71x71x16
        self.fc1 = nn.Linear(80656, 1024)
        self.fc2 = nn.Linear(1024, 120)
        self.fc3 = nn.Linear(120, 3)
        # self.out = nn.Softmax(3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return F.softmax(x, dim=1)
        # output = F.log_softmax(x, dim=1)
        # return torch.exp(output)
        # return output


# with open('./dataset/test/infected/covid/0.jpg', 'rb') as f:
#     # Convert to Numpy array and normalize pixel values by dividing by 255.
#     im = np.asarray(Image.open(f))/255
#     im = transforms.functional.to_tensor(np.array(im)).float()
#     im = im.unsqueeze(0)
#     print(im.shape)
#     target = torch.Tensor([1]).long()
#     print(target.shape)
#     # target = transforms.functional.to_tensor(np.array([1]))


# def train_model(model):
#     model.train()
#     prediction = model(im)
#     # target = model(im)

#     loss = nn.CrossEntropyLoss()
#     output = loss(prediction, target)
#     output.backward()
#     optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
#     optim.step()  # gradient descent

#     print(prediction)
#     # for batch_idx, (images_data, target_labels) in enumerate(train_loader):
#     #     predicted_labels = model(images_data)
#     #     print(predicted_labels)
#     #     print(target_labels)
#     #     # Forced stop
#     #     break


# model_three_class = ModelThreeClass()
# train_model(model_three_class)
