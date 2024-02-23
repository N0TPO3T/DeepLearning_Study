##常见的transforms应用
#输入：PIL，Image.open()。输出：tensor,ToTensor()。作用：narrays,cv.imread()
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs") #定义类
img = Image.open("image/1.jpg") #打开图片
print(img)

# ToTensor的使用
trans_totensor = transforms.ToTensor() #定义类
img_tensor = trans_totensor(img) #调用类中的call函数
writer.add_image("ToTensor", img_tensor) #将其加入到tensorboard中

# Normalize（对图片进行标准化，能够改变灰度值的范围）
trans_norm = transforms.Normalize([1,2,3], [4,5,6])
img_norm = trans_norm(img_tensor)
writer.add_image("Normalize",img_norm)

# Resize 修改图片的形状和大小，非等比缩放
print(img.size)
trans_resize = transforms.Resize((800, 800))
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize, 0)


# Compose -resize -2，compose作用就是能够结合几个不同的操作，如下就是能够直接对PIL图像进行操作，但实际上是【】中的两个操作同时完成
trans_resize_2 = transforms.Resize(512)
# PIL -> PIL -> tenosr 因为resize的数据类型不能是tensor，要是PIL的图片的类型
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize2",img_resize_2)

writer.close()