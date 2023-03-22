import torch
import torch.nn as nn

import warnings
warnings.filterwarnings('ignore')
from resources.code.help_functions import ei_zeichnen
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

class Model2(nn.Module):
    def __init__(self, out):
        super(Model2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = torch.nn.Dropout2d(0.2)
        self.conv2 = nn.Conv2d(32, 32, 2)
        self.fc1 = nn.Linear(4608, 56)
        self.fc2 = nn.Linear(56, out)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = x.view(-1, 32*144)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class Net(nn.Module):
    
    def __init__(self, num_in, num_out):
        
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, 64)
        self.fc5 = nn.Linear(64, num_out)
        

        self.relu=torch.nn.ReLU()

        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        output = self.relu(self.fc1(x))
        output = self.relu(self.fc2(output))
        output = self.relu(self.fc3(output))
        output = self.relu(self.fc4(output))
        output = self.fc5(output)
        output = self.softmax(output)
        return output
    

TRAIN_DATA_PATH = 'DataSet/train'
TEST_DATA_PATH = 'DataSet/test'

train_transforms = transforms.Compose([
  transforms.Resize([64,64]),
  transforms.ToTensor(),
  transforms.Grayscale()
])

train_dataset = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
# net = Net(4096, 3)
net = Model2(3)

optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
loss_func = torch.nn.CrossEntropyLoss()

from tqdm import tqdm

count = 0
total = 0
for x,y in train_dataset:
    total += 1
    # output = net(x.view(-1))
    output = net(x)
    output = torch.unsqueeze(output, 0)
    if torch.argmax(output).item() == y:
        count += 1

print(count, total)
for _ in tqdm(range(5)):
    for x,y in train_dataset:
        y = torch.tensor([y])
        # output = net(x.view(-1))
        output = net(x)
        output = torch.unsqueeze(output, 0)
        loss = loss_func(output, y)
        # print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

count = 0 
total = 0
for x,y in train_dataset:
    total += 1
    # output = net(x.view(-1))
    output = net(x)
    output = torch.unsqueeze(output, 0)
    if torch.argmax(output).item() == y:
        count += 1
        
print(count, total)