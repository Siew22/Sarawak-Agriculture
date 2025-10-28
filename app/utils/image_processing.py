import io
from PIL import Image
import torchvision.transforms as transforms

class ImageProcessor:
    def __init__(self):
        # 这个预处理流程必须与你训练模型时使用的完全一致！
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def process(self, image_bytes: bytes):
        """将原始图片字节流转换为模型所需的Tensor"""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            # 确保图像是RGB格式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return self.transform(image).unsqueeze(0)
        except Exception as e:
            # 可以加入更详细的日志记录
            print(f"Error processing image: {e}")
            raise ValueError("无法处理提供的图像文件，请确保文件未损坏且格式正确。")

# 创建一个全局实例，方便在其他地方调用
image_processor = ImageProcessor()