##下面是线性化层所做的事
import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

class POET(nn.Module):
    def __init__(self):
        super().__init__()
        # infeature就是输入数据的大小，也就是向量形式图片的长度,这个函数的作用就是能够将向量图片的大小转换为想要的图片大小,也可以理解为提取特征
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output

poet = POET()

for data in dataloader:
    imgs, targets = data
    # 下面这步是将图片拉成向量形式，即一行
    # flatten就是把输出展成一行
    output = torch.flatten(imgs)
    output = poet(output)
    print(output.shape)