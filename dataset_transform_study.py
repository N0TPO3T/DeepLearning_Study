import torchvision
from torch.utils.tensorboard import SummaryWriter

# 导入transforms当中的类，也即是经过compose整合的图像处理方法，这里的例子仅仅将PIL图片类型转换为了tensor类型，还可以裁剪之类的
dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

root_dir = "./dataset"


# 构建数据集，transforms就是对数据集的处理方法，也是上文中定义的处理方法
train_set = torchvision.datasets.CIFAR10(root_dir, train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root_dir, train=False, transform=dataset_transform, download=True)


# 用tensorboard的办法将数据集中的文件展示出来
writer = SummaryWriter("logs")

for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()