import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


dataset = torchvision.datasets.CIFAR10('./dataset', train=False, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, 1)

class POET(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )


    def forward(self, x):
        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()
poet = POET()
for data in dataloader:
    imgs, targets = data
    outputs = poet(imgs)
    result_loss = loss(outputs, targets)
    # backward是对loss计算梯度，进行反向传播，就可以计算出每一个结点的参数，从而选择优化器对参数进行优化
    var = result_loss.backward
    print(var)