import torch
import torchvision
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# 确认数据集的位置
dataset = torchvision.datasets.CIFAR10("./dataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)

# 加载数据集到dataloader中，一次取出64个数据
dataloader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(3, 6, 3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

tudui = Tudui()

# print(tudui)结果如下，即tudui这个神经网络包括一个卷积层（可以理解为一次卷积操作），这个卷积层conv2d，输入为3层，输出为6层，也就相当于有两个卷积核（可能不同）（相当于神经元的数量）对图片进行操作，以增加网络的复杂度
#Tudui((conv1): Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1)))


# ()中为日志的保存路径
writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    print(imgs.shape)
    print(output.shape)

    # torch.Size([64, 3, 32, 32])
    writer.add_images("input", imgs, step)

    # torch.Size([64, 6, 30, 30]) -> [xxx,3,30,30],因为tensorborad打不开6通道的图片，所以把6通道拆开为两个3通道进行显示，-1代表自动进行计算而不指定值的大小
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)

    step += 1

writer.close()