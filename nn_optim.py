import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


dataset = torchvision.datasets.CIFAR10('./dataset', train=False, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, 64)

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


# CrossEntropyLoss是一个交叉熵损失函数
loss = nn.CrossEntropyLoss()
poet = POET()
#SGD是随机梯度下降优化器,第一个数输入params是我们模型的参数，lr是学习速率，
optim = torch.optim.SGD(poet.parameters(), lr=0.01)

for epoch in range(20):
    running_loss = 0.0
#以下是对整个数据集进行一轮学习，所以效果很差
    for data in dataloader:
        imgs, targets = data
        outputs = poet(imgs)
        result_loss = loss(outputs, targets)
        # 将每一个可调参数的梯度设置为0
        optim.zero_grad()
        # backward是对loss计算梯度，进行反向传播，就可以计算出每一个结点的参数，从而选择优化器对参数进行优化
        result_loss.backward()
        # 开始使用优化器optim对梯度进行调优
        optim.step()
        print(result_loss)
        # 此处的result_loss是指在一轮训练中，优化器总的梯度
        running_loss = running_loss + result_loss
    print(running_loss)