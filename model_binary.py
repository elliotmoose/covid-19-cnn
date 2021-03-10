import torch
from lung_dataset import Lung_Dataset
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms


class Model_binary_1(nn.Module):
    def __init__(self):
        super(Model_binary_1, self).__init__()
        # input img = 150x150
        # Conv2D: 1 input channel, 16 output channels, 7 kernel, stride of 1. -> 144x144x16
        self.conv1 = nn.Conv2d(1, 16, 7, 1)

        # Conv2D: 4 input channel, 8 output channels, 5 by 5 kernel, stride of 2.
        # self.conv2 = nn.Conv2d(8, 16, 5, 1)

        # maxpooling -> 48x48x16
        self.pool = nn.MaxPool2d(3, 3)

        self.conv3 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv4 = nn.Conv2d(16, 16, 3, 1, 1)

        self.conv5 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv6 = nn.Conv2d(16, 16, 3, 1, 1)

        self.conv7 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv8 = nn.Conv2d(16, 16, 3, 1, 1)

        self.fc1 = nn.Linear(36864, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        residual = x
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x += residual

        residual = x
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x += residual

        residual = x
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x += residual

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        output = F.softmax(x, dim=1)
        return output


model = Model_binary_1()
print(model)

ld_train = Lung_Dataset("train", "binary")

batch_size = 512


train_loader = DataLoader(ld_train, batch_size=batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


for epoch in range(50):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        # running_loss += loss.item()
        print(loss.item())
        # print(loss.item())
        # if i % 10 == 9:    # print every 2000 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / 9))
        #     running_loss = 0.0

    # print(f'Epoch: {i} Loss: {running_loss}')


print('Finished Training')
