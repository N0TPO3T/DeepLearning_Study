import torch
from torch import nn


# 定义神经网络的模板 super()函数用于在子类中调用父类
class Tudui(nn.Module):
    def __init__(self) -> object:
        # 重载init函数
        super().__init__()

    # __call__中调用了forward函数，故自动运行
    def forward(self, input):
        output = input + 1
        return output


tudui = Tudui()
# tensor函数用于生成张量
x = torch.tensor(1.0)
output = tudui(x)
print(output)