from glob import glob
import numpy as np
import torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
sys.path.append("/zooper1/fontbakers/jped/font-bakers/serialization")
from fontDataset import Dataset

DATA_PATH = '/zooper1/fontbakers/protos/protoFiles'
DIMENSIONS = (20, 30, 3, 2)
TRAIN_TEST_SPLIT = 0.8
NUM_EPOCHS = 2
CHARS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
    'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
    't', 'u', 'v', 'w', 'x', 'y', 'z', 'zero', 'one', 'two', 'three', 'four',
    'five', 'six', 'seven', 'eight', 'nine', 'exclam', 'numbersign', 'dollar',
    'percent', 'ampersand', 'asterisk', 'question', 'at'
]

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

trainset = Dataset(DATA_PATH, DIMENSIONS)
trainloader = data.DataLoader(
    trainset, batch_size=1, shuffle=True, num_workers=1)


class Net(nn.Module):
    def __init__(self):
        # Input shape: (93, 3, 2, 80)
        super(Net, self).__init__()
        self.conv1 = nn.Conv3d(20, 32, [1, 2, 2], padding=[1, 2, 3])
        self.conv2 = nn.Conv3d(32, 64, [1, 2, 2])
        self.pool = nn.MaxPool3d([2, 2, 2])
        self.conv3 = nn.Conv3d(64, 64, [1, 2, 2], padding=[1, 1, 1])
        self.conv4 = nn.Conv3d(64, 64, [1, 2, 2])
        self.fc1 = nn.Linear(6912, len(CHARS))

    def forward(self, x):
        x = torch.squeeze(x)
        x = x.double().cuda(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.pool(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        return x


# Compile
net = Net().double().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

print('Starting training.')

labelsToNums = {label: x for x, label in enumerate(CHARS)}

for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for i, f in enumerate(trainloader):  # font.shape = (1, 93, 80, 3, 2)
        font, labels = f
        labels = labelsToNums[labels[0]]
        labels = torch.tensor(labels).repeat(font.size()[1]).cuda(device)
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward, backward, optimize
        prediction = net(font)
        loss = criterion(prediction, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 1000 == 999:  # Print every 1000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1,
                                            running_loss / 2000))
            running_loss = 0.0

print('Finished training.')
print('Computing test accuracy.')

correct = 0
total = 0
with torch.no_grad():
    for font in testloader:
        font = torch.transpose(torch.transpose(font.squeeze(), 1, 2), 2, 3)
        font = font.to(device)
        outputs = net(font)
        _, predicted = torch.max(outputs.data, 1)
        total += true_labels.size()[0]
        correct += (predicted == true_labels).sum().item()

print('Accuracy of network on test fonts: %d %%' % (100 * correct / total))
