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
        conv1_output_channels = 8
        conv2_output_channels = 16
        conv3_output_channels = 24
        conv4_output_channels = 24
        conv5_output_channels = 32
        conv6_output_channels = 64

        # conv1_output_size = (inp_img_size-kernel_size+2*padding)/stride + 1;
        #150x150x1
        self.conv1 = nn.Conv2d(1, conv1_output_channels, kernel_size, stride) #148x148x8
        self.conv2 = nn.Conv2d(conv1_output_channels, conv2_output_channels, kernel_size, stride) #146x146x16        
        self.pool = nn.MaxPool2d(2, 2) 
        #73x73x16
        self.conv3 = nn.Conv2d(conv2_output_channels, conv3_output_channels, kernel_size, stride, padding=1) #73x73x16
        self.conv4 = nn.Conv2d(conv3_output_channels, conv4_output_channels, kernel_size, stride, padding=1) #73x73x16
        self.skip_conv1 = nn.Conv2d(conv2_output_channels, conv4_output_channels, kernel_size, stride, padding=1) #146x146x16
        #pool #37x37x16
        self.conv5 = nn.Conv2d(conv4_output_channels, conv5_output_channels, kernel_size, stride, padding=1) #35x35x16
        self.conv6 = nn.Conv2d(conv5_output_channels, conv6_output_channels, kernel_size, stride, padding=1) #33x33x16
        self.skip_conv2 = nn.Conv2d(conv4_output_channels, conv6_output_channels, kernel_size, stride, padding=1) #146x146x16
        # self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        #pool 17*17*16        
        self.fc1 = nn.Linear(5184, 1024)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(512, 3)
        # self.out = nn.Softmax(3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        residual = self.skip_conv1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x += residual
        x = self.pool(x)
        
        # residual = x
        residual = self.skip_conv2(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x += residual
        x = self.pool(x)

        x = self.pool(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1) 
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))        
        x = self.dropout2(x)
        x = self.fc3(x)
        # return F.softmax(x, dim=1)
        return F.log_softmax(x, dim=1)
        # return torch.exp(output)
        # return output
