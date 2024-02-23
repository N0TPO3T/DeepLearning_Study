import torchvision 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备测试数据集
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())

# batch_size代表每一次抓取数据的个数
# epoch为true时，代表每次读取数据都是随机的
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)


# 测试数据集中第一张图片
img, target = test_data[0]
print(img.shape)
print(target)


writer = SummaryWriter("logs")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        writer.add_images(f"Epoch:{epoch}", imgs, step)
        step += 1


writer.close()