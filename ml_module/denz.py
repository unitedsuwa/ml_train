import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from pyprind import prog_bar
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST

EPOCH = 3

MINIST_DATA_PATH = '/tmp/mnist'
CIAFR10_DATA_PATH = '/tmp/cifar10'

USE_CUDA =  torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

print('Use device is', str(DEVICE).upper())


class DatasetBuilder:

    @staticmethod
    def mnist_dataset(data_path):
        train_data = MNIST(data_path, train=True, download=True,
                        transform=transforms.ToTensor())
        train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

        test_data = MNIST(data_path, train=False, download=True,
                        transform=transforms.ToTensor())
        test_loader = DataLoader(test_data, batch_size=4, shuffle=False)



        return train_loader, test_loader, 28*28

    @staticmethod
    def cifer_dataset(data_path):
        train_data = CIFAR10(data_path, train=True, download=True,
                   transform=transforms.ToTensor())
        train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

        test_data = CIFAR10(data_path, train=False, download=True,
                        transform=transforms.ToTensor())
        test_loader = DataLoader(test_data, batch_size=4, shuffle=False)

        return train_loader, test_loader, 32*32*3


class DenzNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, 50)
        self.l2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.l1(x)
        x = self.l2(x)
        return x


class Train:

    def __init__(self, train_loader, test_loader, input_size, epoch):
        self.train_dataset = train_loader
        self.test_dataset = test_loader
        self.train_total = len(self.train_dataset.dataset)
        self.test_total = len(self.test_dataset.dataset)
        self.net = DenzNet(input_size).to(DEVICE)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        self.epochs = epoch



    def train(self, epoch):
        running_loss = 0.0
        print('Start  Train epoch: {}/{}'.format(epoch + 1, self.epochs))
        for index, data in enumerate(self.train_dataset):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            self.optimizer.zero_grad()
            outputs = self.net(inputs)

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            if index % 5000  == 4999:
                print('Finished({}/{})  loss: {:.3f}'.format(index + 1, len(self.train_dataset), running_loss / 1000))
                running_loss = 0.0

    def test(self, epoch):
        print('Start Test')
        correct = 0
        test_loss = 0.0
        with torch.no_grad():
            for data in prog_bar(self.test_dataset):
                inputs, labels = data
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = self.net(inputs)
                test_loss += self.criterion(outputs, labels).sum().item()
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()

            print('epoch:{}  '.format(epoch + 1) + 'Avarage Loss: {:.4f}  Accuracy: {}/{} = {:.4f}%'.format(test_loss / 1000, correct, self.test_total, 100 * correct / self.test_total))
        print('Total Accuracy: {}/{} = {:.6f}%'.format(correct, self.test_total, 100 * correct / self.test_total))


    def run(self):
        for epoch in range(self.epochs):
            print(f"Epoch: {epoch}")
            self.train(epoch)
            self.test(epoch)


def view_samples(loader):
    # classes = ('plane', 'car', 'bird', 'cat',
    #        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    dataiter = iter(loader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images))
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    print(' '.join('%5s' % labels[j] for j in range(4)))



def main():
    train_dataset, test_dataset, input_size = DatasetBuilder.mnist_dataset(MINIST_DATA_PATH)
    # train_dataset, test_dataset, input_size = DatasetBuilder.cifer_dataset(CIAFR10_DATA_PATH)
    # view_samples(train_dataset)

    trainer = Train(train_dataset, test_dataset, input_size, EPOCH)
    trainer.run()

if __name__ == "__main__":
    main()
