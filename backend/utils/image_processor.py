import base64
import io
import numpy as np
from PIL import Image
import cv2


class ImageProcessor:
    """图像处理工具类"""
    
    def __init__(self):
        pass
    
    def decode_base64_image(self, base64_string):
        """
        解码base64图像
        
        Args:
            base64_string: base64编码的图像字符串
        
        Returns:
            PIL Image
        """
        try:
            # 移除data URL前缀（如果存在）
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # 解码
            image_data = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_data))
            
            # 转换为RGB（如果是RGBA等格式）
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image: {e}")
    
    def encode_image_to_base64(self, image):
        """
        编码图像为base64
        
        Args:
            image: PIL Image或numpy array
        
        Returns:
            base64编码的字符串
        """
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # 保存到内存
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=95)
            buffer.seek(0)
            
            # 编码为base64
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            
            return f"data:image/jpeg;base64,{image_base64}"
            
        except Exception as e:
            raise ValueError(f"Failed to encode image to base64: {e}")
    
    def crop_face(self, image, box, margin=0.2):
        """
        裁剪人脸区域（带边距）
        
        Args:
            image: PIL Image
            box: [x1, y1, x2, y2]
            margin: 边距比例
        
        Returns:
            PIL Image
        """
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            x1, y1, x2, y2 = map(int, box)
            
            # 添加边距
            w = x2 - x1
            h = y2 - y1
            margin_w = int(w * margin)
            margin_h = int(h * margin)
            
            # 扩展边界框
            x1 = max(0, x1 - margin_w)
            y1 = max(0, y1 - margin_h)
            x2 = min(image.width, x2 + margin_w)
            y2 = min(image.height, y2 + margin_h)
            
            # 裁剪
            face = image.crop((x1, y1, x2, y2))
            
            return face
            
        except Exception as e:
            raise ValueError(f"Failed to crop face: {e}")
    
    def resize_image(self, image, size=(160, 160)):
        """
        调整图像大小
        
        Args:
            image: PIL Image
            size: (width, height)
        
        Returns:
            PIL Image
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        return image.resize(size, Image.BILINEAR)
    
    def preprocess_for_model(self, image, size=(160, 160)):
        """
        预处理图像用于模型输入
        
        Returns:
            numpy array
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # 调整大小
        image = image.resize(size)
        
        # 转换为numpy array并归一化
        img_array = np.array(image).astype(np.float32)
        img_array = (img_array - 127.5) / 128.0
        
        return img_array
    
    def draw_boxes(self, image, boxes, names=None, confidences=None):
        """
        在图像上绘制人脸框和标签
        
        Args:
            image: PIL Image或numpy array
            boxes: list of [x1, y1, x2, y2]
            names: list of names (optional)
            confidences: list of confidence scores (optional)
        
        Returns:
            numpy array (BGR format for display)
        """
        try:
            # 转换为numpy array
            if isinstance(image, Image.Image):
                img = np.array(image)
            else:
                img = image.copy()
            
            # 转换为BGR（OpenCV格式）
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                
                # 绘制矩形
                color = (0, 255, 0) if names and names[idx] != 'Unknown' else (0, 0, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签
                if names:
                    label = names[idx]
                    if confidences and idx < len(confidences):
                        label += f" ({confidences[idx]:.2f})"
                    
                    # 绘制标签背景
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    
                    # 绘制文字
                    cv2.putText(img, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return img
            
        except Exception as e:
            print(f"Error drawing boxes: {e}")
            return image
    
    def normalize_image(self, image):
        """归一化图像"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        return image.astype(np.float32) / 255.0
    
    def augment_image(self, image, rotation=0, flip=False):
        """
        数据增强
        
        Args:
            image: PIL Image
            rotation: 旋转角度
            flip: 是否水平翻转
        
        Returns:
            PIL Image
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if rotation != 0:
            image = image.rotate(rotation)
        
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        return image