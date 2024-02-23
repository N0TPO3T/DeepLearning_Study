import torch
import torch.nn.functional as F

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

kernal = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

# reshape将数据转换为所需要的维度，因为卷积函数的输入是要求四输入维度的
input = torch.reshape(input, (1, 1, 5, 5))
kernal = torch.reshape(kernal, (1, 1, 3, 3))

print(input.shape)
print(kernal.shape)
# torch.Size([1, 1, 3, 3])这是结果，分别为batch个数（一次处理的图像数目），输入的通道数，长，宽

# conv2d是二维卷积处理
output = F.conv2d(input, kernal, stride=1)
print(output)

output2 = F.conv2d(input, kernal, stride=2)
print(output2)

output3 = F.conv2d(input, kernal, stride=1, padding=1)
print(output3)