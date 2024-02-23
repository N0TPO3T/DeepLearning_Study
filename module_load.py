import torch
import torchvision

# 方式1 -》保存方式1， 加载模型
model = torch.load("vgg16_method1.pth")
#print(model)

# 方式2，加载模型,保存的是模型中的参数，输出为参数，还需还原成模型结构
vgg16 = torchvision.models.vgg16(pretrained=False)
# 下面的函数就是加载模型的参数为字典
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
#model = torch.load("vgg16_method2.pth")
print(vgg16)

#针对方式1的陷阱 可以直接import模型文件
import torch
from module_save import *  # 它就相当于把 model_save.py 里的网络模型定义写到这里了
# tudui = Tudui # 不需要写这一步，不需要创建网络模型    
model = torch.load("tudui_method1.pth")
print(model)