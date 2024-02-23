# 常用的非线性激活函数ReLU和Sigmoid（一般用于二分类输出层）
# 非线性变换使网络变成通用函数逼近器，如果不引入非线性特征，网络就只能模拟线性函数的特征，变得无意义
# ReLU的作用就是小于零的置为0，大于零的不变
import torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, 64)

class Poet(nn.Module):
    def __init__(self):
        super().__init__()
        # inplace函数就是是否将输入进行保存，如果是true则直接将input的值替换，一般才用flase保存原始数据
        self.relu = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output

step = 0
# 导入Poet类
poet = Poet()
writer = SummaryWriter("logs")
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = poet(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()