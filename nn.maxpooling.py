# pooling池化操作，即对输入进行降维，比如maxpooling就是按照kernel的大小取其中的最大值，每次移动的步长即为池化核的大小1

# 池化的目的也就是减小数据量

import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, download=True, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, 64)

#input = torch.tensor([[1, 2, 0, 3, 1],[0, 1, 2, 3, 1],[1, 2, 1, 0, 0],[5, 2, 3, 1, 1],[2, 1, 0, 1, 1]], dtype=torch.float)
#nput = torch.reshape(input, [-1, 1, 5, 5])
#print(input.shape)

class Poet(nn.Module):
    def __init__(self):
        super().__init__()
        # kernel代表池化核的大小，ceilmode代表池化的模式，true代表保留不完全的池化对象，false代表舍弃
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

poet = Poet()

writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    # 池化后的图片不会出现多个channel
    output = poet(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()