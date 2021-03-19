from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

class ModelThreeClassLite(nn.Module):
    def __init__(self):
        super(ModelThreeClassLite, self).__init__()

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


from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

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
from tqdm import tqdm
from datetime import datetime
import time


def validate_model(model, dataloader, criterion, device):
  running_loss = 0
  correct_count = 0
  for images, labels in dataloader:
    images, labels = images.to(device), labels.to(device)    
    prediction = model(images)
    loss = criterion(prediction, labels).item()
    running_loss += loss
    equality = (labels.data == prediction.max(dim=1)[1])
    correct_count += equality.type(torch.FloatTensor).sum()
  accuracy = correct_count/len(dataloader.dataset)
  running_loss /= len(dataloader)
  return running_loss, accuracy

def train_model(model, dataloader, criterion, epochs=50):
    model.train()

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    start_time = datetime.now()
    best_val_accuracy = 0
    best_val_loss = 999
    best_weights = None

    for i in range(epochs):
        running_loss = 0

        model.train()
        with tqdm(dataloader, position=0, leave=False) as progress_bar:          
          for images_data, target_labels in progress_bar:
              images_data = images_data.to(device)
              target_labels = target_labels.to(device)
              
              optim.zero_grad()
              model.zero_grad()
              
              prediction = model(images_data)              
              loss = criterion(prediction, target_labels)
              loss.backward()            
              optim.step()
              running_loss += loss.item()
        running_loss /= len(dataloader)
        trainLossGraph.append(running_loss)
        datetime.now()
        time_elapsed = (datetime.now() - start_time)

        model.eval()
        val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)
          
        valLossGraph.append(val_loss)
        valAccuracyGraph.append(val_accuracy)
        progress_bar.refresh()

        tqdm.write(f'\n===Epoch: {i}===')
        tqdm.write(f'== Loss: {running_loss:.3f} Time: {datetime.now()} Elapsed: {time_elapsed}')    
        tqdm.write(f'== Val Loss: {val_loss:.3f} Val Accuracy: {val_accuracy:.3f}')    

        # if val_loss < best_val_loss or val_accuracy >= best_val_accuracy:
        #   print("=== NEW BEST!! ===")
        #   best_weights = model.state_dict()            
        #   best_val_loss = val_loss
        #   best_val_accuracy = val_accuracy
        
        time.sleep(0.4)
    
    print(f"best accuracy: {best_val_accuracy}")
    return model.state_dict()    
    # return best_weights    
# device = torch.device('cuda:0')

# train_ld = Lung_Dataset('train', 'three_class')#, augmentation='hist_equal'
# val_ld = Lung_Dataset('val', 'three_class')#, augmentation='hist_equal'
# test_ld = Lung_Dataset('test', 'three_class')#, augmentation='hist_equal'
# train_loader = DataLoader(train_ld, batch_size=512, shuffle=True)
# val_loader = DataLoader(val_ld, batch_size=512, shuffle=True)
# test_loader = DataLoader(test_ld)

# trainLossGraph = []
# valLossGraph = []
# valAccuracyGraph = []

# model_three_class_lite = ModelThreeClassLite()
# model_three_class_lite = model_three_class_lite.to(device)
# model_three_class_lite.train()
# criterion = nn.NLLLoss()
# best_weights = train_model(model_three_class_lite, train_loader, criterion, epochs=30)
# # print(torch.cuda.is_available())

# model_three_class_lite.load_state_dict(best_weights)
# model_three_class_lite.eval()
# criterion = nn.CrossEntropyLoss()

# test_loss, test_accuracy = validate_model(model_three_class_lite, test_loader, criterion, device)
# print('test:', test_accuracy)

# train_loss, train_accuracy = validate_model(model_three_class_lite, train_loader, criterion, device)
# print('train:', train_accuracy)