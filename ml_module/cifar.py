import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from pyprind import prog_bar
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

epochs = 50


data_path = '/tmp/cifar10'
# cifar_data = CIFAR10(data_path, train=True,
#                    download=True, transform=transforms.ToTensor())
# data_loader = DataLoader(cifar_data, batch_sampler=4, shuffle=False)

use_cuda =  torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
torch.backends.cudnn.benchmark=True


print('Use device is', str(device).upper())

# def data_show():
#     data_iter = iter(data_loader)
#     images, labels = data_iter.next()

#     npimg = images[0].np()
#     npimg = npimg.reshape((28, 28))
#     plt.imshow(npimg, cmap='gray')
#     print('label:', labels[0])


train_data = CIFAR10(data_path, train=True, download=True,
                   transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, **kwargs)

test_data = CIFAR10(data_path, train=False, download=True,
                  transform=transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=1000, shuffle=False, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(32*32*3, 1200)
        self.l2 = nn.Linear(1200, 600)
        self.l3 = nn.Linear(600, 10)
        self.dropout1 = nn.Dropout2d(0.2)
        self.dropout2 = nn.Dropout2d(0.2)

    def forward(self, x):
        x = x.view(-1, 32*32*3)
        x = F.relu(self.l1(x))
        x = self.dropout1(x)
        x = F.relu(self.l2(x))
        x = self.dropout2(x)
        x = F.relu(self.l3(x))
        return x


net = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

train_total = len(train_loader)
test_total = len(test_loader.dataset)

interval = train_total // 3

def train():
    running_loss = 0.0
    print('Start  Train epoch: {}/{}'.format(epoch + 1, epochs))
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1)  % interval  == 0 :
            print('Finished({}/{})  loss: {:.3f}'.format(i * len(inputs), len(train_loader.dataset), running_loss / interval))
            running_loss = 0.0

def test():
    print('Start Test')
    correct = 0
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in prog_bar(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            test_loss += criterion(outputs, targets).sum().item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(targets.view_as(pred)).sum().item()

        print('epoch:{}  '.format(epoch + 1) + 'Avarage Loss: {:.4f}  Accuracy: {}/{} = {:.2f}%'.format(test_loss / 1000, correct, test_total, 100 * correct / test_total))


for epoch in range(epochs):
    train()
    test()
