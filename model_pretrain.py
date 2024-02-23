import torchvision
from torch import nn

# 此时是加载网络模型，即参数为默认的参数,该网络输出1000个类别
vgg16_false = torchvision.models.vgg16(pretrained=False)
# 此时要从网络上下载pretrained的数据集，参数不同-
#vgg16_true = torchvision.models.vgg16(pretrained=True)

print(vgg16_false)

# 这个数据集有十分类，故上面的1000分类的模型用不了
train_data = torchvision.datasets.CIFAR10('./dataset', train=True, transform=torchvision.transforms.ToTensor(), download=True)

# 利用现有网络对十分类数据集进行处理，（迁移学习）,在现有网络中添加一层(直接在vgg中加）
#vgg16_false.add_module('add_linear', nn.Linear(1000, 10))
#print(vgg16_false)

# 在vgg的分类器中添加一层
vgg16_false.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_false)