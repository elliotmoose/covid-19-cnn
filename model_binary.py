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
        # Conv2D: 1 input channel, 4 output channels, 3 by 3 kernel, stride of 1.
        self.conv1 = nn.Conv2d(1, 4, 3, 1)
        # Conv2D: 4 input channel, 8 output channels, 5 by 5 kernel, stride of 2.
        self.conv2 = nn.Conv2d(4, 8, 5, 2)
        self.fc1 = nn.Linear(41472, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.softmax(x, dim=1)
        return output


model = Model_binary_1()
print(model)

ld_train = Lung_Dataset("train", "binary")

batch_size = 8


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
        running_loss += loss.item()
        # print(loss.item())
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

    # print(f'Epoch: {i} Loss: {running_loss}')


print('Finished Training')
