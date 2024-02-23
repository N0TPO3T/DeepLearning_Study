from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
#transforms文件是一个工具箱，一般用于图片处理，例如转换为tensor文件或者reszie图片
#tensor的数据类型讲解，通过transforms.ToTensor去解决两个问题：1.transfomrs的使用，2.Tensor数据类型的特点:神经网络中的各种参数，比如回馈系数等

##以下为先用cv2或者PIL方式将图片打开，这时图片为NUMPY或者其他格式，再用tensor将其转换为Tenosr形式并在tensorborad上显示的过程
#-------------------------------------------------------------------------
img_path = "data/train/ants_image/0013035.jpg"
img = Image.open(img_path)

#导入SummarWriter类
writer = SummaryWriter("logs")

#导入transforms.TOTesor类
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

print(tensor_img)

writer.add_image("Tensor_img", tensor_img)

writer.close()
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————