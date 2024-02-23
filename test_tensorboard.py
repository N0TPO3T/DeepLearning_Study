from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

# 创建一个储存生成文件的文件夹“log”
writer = SummaryWriter("logs")

# 打开图片,类型受限，需要时tensor型或者numpy型，用cv2读取的类型就是numpy型图片,但是注意add_image函数默认的图像类型是CHW型，而我们读取的是HWC型，故需要制定图像类型
image_path = "data/train/ants_image/0013035.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)

writer.add_image("test",img_array,1,dataformats='HWC')
# y = x 生成一个y=2x的图像
for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)

writer.close()



# 打开tensorboard的命令为tensorboard --logdir=logs 
