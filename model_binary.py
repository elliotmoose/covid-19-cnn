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
    def __init__(self):
        super(Model_binary_bn, self).__init__()
        
        # input img = 150x150
        # First block of Residual Block (3x3x16/2) + Pooling(2x2/2), output size = 19x19x16
        self.conv1 = nn.Conv2d(1, 16, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(16, 16, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv_skip1 = nn.Conv2d(1, 16, 1, 4)
        self.bn_skip1 = nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.pool1 = nn.MaxPool2d(2, 2)
        # Second block of Residual Block (3x3x32/1) + Pooling(2x2/2), output size = 9x9x32
        self.conv3 = nn.Conv2d(16, 32, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv_skip2 = nn.Conv2d(16, 32, 1, 1)
        self.bn_skip2 = nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.pool2 = nn.MaxPool2d(2, 2)
        # Third block of Residual Block (3x3x32/1) + Global Average Pooling, output size = 64
        self.conv5 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv6 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv_skip3 = nn.Conv2d(32, 64, 1, 1)
        self.bn_skip3 = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        # FC layer
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(32, 2)


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
        x = self.gap(x)
        x = torch.flatten(x, 1)
        # FC
        x = self.dropout1(self.fc1(x)) 
        x = self.dropout2(self.fc2(x)) 

        output = F.log_softmax(x, dim=1)
        return output

def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
                
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy

def test(model, testloader, device='cuda'):  
    model.to(device)
    accuracy = 0
    with torch.no_grad():
        model.eval()
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
                    
            output = model.forward(images)
            
            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
        
        print('Testing Accuracy: {:.3f}'.format(accuracy/len(testloader)))


def train(model, model_name, batch_size, n_epochs, lr, train_loader, val_loader, saved_model_path, device = "cuda", print_every=40, save_every=50):
    input_sample, _ =  next(iter(train_loader))
    print(summary(model, tuple(input_sample.shape[1:]), device=device))

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr= lr) 

    train_loss_ls = []
    val_loss_ls = []
    
    steps =0
    running_loss = 0.0
    for e in range(n_epochs):  # loop over the dataset multiple times

        # Training
        model.train()
    #     train_loader1 = DataLoader(ld_train1, batch_size=batch_size, shuffle=True)
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            steps += 1

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if steps % save_every == 0:
                filepath = saved_model_path + f"{model_name}-b{batch_size}-e{e}-step{steps}.pt"
                torch.save(model, filepath)

            if steps % print_every == 0:
                # Eval mode for predictions
                model.eval()

                # Turn off gradients for validation
                with torch.no_grad():
                    test_loss, accuracy = validation(model, val_loader, criterion, device)

                print("Epoch: {}/{} - ".format(e+1, n_epochs),
                      "Training Loss: {:.3f} - ".format(running_loss/print_every),
                      "Validation Loss: {:.3f} - ".format(test_loss/len(val_loader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(val_loader)))
                
                train_loss_ls.append(running_loss/print_every)
                val_loss_ls.append(test_loss/len(val_loader))
                running_loss = 0
                

                # Make sure training is back on
                model.train()

    print("Finished training")
    plt.plot(train_loss_ls, label = "train_loss")
    plt.plot(val_loss_ls, label = "val_loss")
    plt.legend()
    plt.savefig(saved_model_path+'train_val_loss.png')
    plt.show()
    return model
