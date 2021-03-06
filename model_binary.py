import os
import torch
import time
from lung_dataset import Lung_Dataset
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from matplotlib import pyplot as plt
from torchsummary import summary
from tqdm import tqdm
from datetime import datetime

class Model_binary(nn.Module):
    def __init__(self):
        super(Model_binary, self).__init__()
        
        # input img = 150x150
        # First block of Residual Block (3x3x16/2) + Pooling(2x2/2), output size = 19x19x16
        self.conv1 = nn.Conv2d(1, 16, 3, 2, 1)
        self.conv2 = nn.Conv2d(16, 16, 3, 2, 1)
        self.conv_skip1 = nn.Conv2d(1, 16, 1, 4)
        self.pool1 = nn.MaxPool2d(2, 2)
        # Second block of Residual Block (3x3x32/1) + Pooling(2x2/2), output size = 9x9x32
        self.conv3 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv_skip2 = nn.Conv2d(16, 32, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        # Third block of Residual Block (3x3x32/1) + Global Average Pooling, output size = 64
        self.conv5 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv6 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_skip3 = nn.Conv2d(32, 64, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        # FC layer
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(32, 2)


    def forward(self, x):
        # Block 1
        residual1 = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        residual1 =  F.relu(self.conv_skip1(residual1))
        x += residual1
        x = self.pool1(x)
        
        # Block 2
        residual2 = x
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        residual2 =  F.relu(self.conv_skip2(residual2))
        x += residual2
        x = self.pool2(x)
        
        # Block 3
        residual3 = x
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        residual3 =  F.relu(self.conv_skip3(residual3))
        x += residual3
        x = self.gap(x)
        x = torch.flatten(x, 1)
        # FC
        x = self.dropout1(self.fc1(x)) 
        x = self.dropout2(self.fc2(x)) 

        output = F.log_softmax(x, dim=1)
        return output
    
    
class Model_binary_bn(nn.Module):
    def __init__(self, classifier):
        super(Model_binary_bn, self).__init__()
        self.classifier = classifier
        
        # input img = 150x150
        # First block of Residual Block (3x3x16/2) + Pooling(2x2/2), output size = 19x19x16
        self.conv1 = nn.Conv2d(1, 16, 3, 2, 1)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(16, 16, 3, 2, 1)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv_skip1 = nn.Conv2d(1, 16, 1, 4)
        nn.init.xavier_uniform_(self.conv_skip1.weight)
        self.bn_skip1 = nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.pool1 = nn.MaxPool2d(2, 2)
        # Second block of Residual Block (3x3x32/1) + Pooling(2x2/2), output size = 9x9x32
        self.conv3 = nn.Conv2d(16, 32, 3, 1, 1)
        nn.init.xavier_uniform_(self.conv3.weight)
        self.bn3 = nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4 = nn.Conv2d(32, 32, 3, 1, 1)
        nn.init.xavier_uniform_(self.conv4.weight)
        self.bn4 = nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv_skip2 = nn.Conv2d(16, 32, 1, 1)
        nn.init.xavier_uniform_(self.conv_skip2.weight)
        self.bn_skip2 = nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.pool2 = nn.MaxPool2d(2, 2)
        # Third block of Residual Block (3x3x32/1) + Global Average Pooling, output size = 64
        self.conv5 = nn.Conv2d(32, 64, 3, 1, 1)
        nn.init.xavier_uniform_(self.conv5.weight)
        self.bn5 = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv6 = nn.Conv2d(64, 64, 3, 1, 1)
        nn.init.xavier_uniform_(self.conv6.weight)
        self.bn6 = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv_skip3 = nn.Conv2d(32, 64, 1, 1)
        nn.init.xavier_uniform_(self.conv_skip3.weight)
        self.bn_skip3 = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        if self.classifier == 2:
            self.pool3 = nn.MaxPool2d(2, 2)
            # Forth block of Residual Block (3x3x32/1) + Global Average Pooling, output size = 64
            self.conv7 = nn.Conv2d(64, 128, 3, 1, 1)
            nn.init.xavier_uniform_(self.conv7.weight)
            self.bn7 = nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            self.conv8 = nn.Conv2d(128, 128, 3, 1, 1)
            nn.init.xavier_uniform_(self.conv8.weight)
            self.bn8 = nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            self.conv_skip4 = nn.Conv2d(64, 128, 1, 1)
            nn.init.xavier_uniform_(self.conv_skip4.weight)
            self.bn_skip4 = nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        # FC layer
        if self.classifier == 1:
            self.dropout1 = nn.Dropout(p=0.3)
            self.fc1 = nn.Linear(64, 32)
            nn.init.xavier_uniform_(self.fc1.weight)
            self.fc2 = nn.Linear(32, 2)
            nn.init.xavier_uniform_(self.fc2.weight)
        else:
            self.fc1 = nn.Linear(128, 64)
            nn.init.xavier_uniform_(self.fc1.weight)
            self.fc2 = nn.Linear(64, 2)
            nn.init.xavier_uniform_(self.fc2.weight)

        
    def forward(self, x):
        # Block 1
        residual1 = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        residual1 =  F.relu(self.bn_skip1(self.conv_skip1(residual1)))
        x += residual1
        x = self.pool1(x)
        
        # Block 2
        residual2 = x
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        residual2 =  F.relu(self.bn_skip2(self.conv_skip2(residual2)))
        x += residual2
        x = self.pool2(x)
        
        # Block 3
        residual3 = x
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        residual3 =  F.relu(self.bn_skip3(self.conv_skip3(residual3)))
        x += residual3
        
        if self.classifier == 2:
            x = self.pool3(x)
            # Block 4
            residual4 = x
            x = F.relu(self.bn7(self.conv7(x)))
            x = F.relu(self.bn8(self.conv8(x)))
            residual4 =  F.relu(self.bn_skip4(self.conv_skip4(residual4)))
            x += residual4
        
        x = self.gap(x)
        x = torch.flatten(x, 1)
        # FC
        x = self.fc1(x)
        if self.classifier == 1:
            x = self.dropout1(x) 
        x = self.fc2(x) 

        output = F.log_softmax(x, dim=1)
        return output
