import numpy as np
from PIL import Image
import torch


class FaceDetector:
    """人脸检测器（使用MTCNN）"""
    
    def __init__(self, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = None
        self._load_detector()
    
    def _load_detector(self):
        """加载MTCNN检测器"""
        try:
            # 使用项目中的MTCNN
            from nnmodels.mtcnn import MTCNN
            
            self.mtcnn = MTCNN(
                image_size=160,
                margin=20,
                min_face_size=40,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=False,
                device=self.device,
                keep_all=True
            )
            print("MTCNN loaded successfully")
            
        except Exception as e:
            print(f"Warning: Could not load MTCNN: {e}")
            print("Using simplified face detector")
            self.mtcnn = None
    
    def detect_faces(self, image):
        """
        检测图像中的所有人脸
        
        Args:
            image: PIL Image或numpy array
        
        Returns:
            list of boxes: [[x1, y1, x2, y2], ...]
        """
        try:
            # 转换为PIL Image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            if self.mtcnn is not None:
                # 使用MTCNN检测
                boxes, probs = self.mtcnn.detect(image)
                
                if boxes is not None:
                    # 过滤低置信度检测
                    valid_boxes = []
                    for box, prob in zip(boxes, probs):
                        if prob > 0.9:  # 置信度阈值
                            valid_boxes.append(box)
                    
                    return np.array(valid_boxes) if valid_boxes else np.array([])
                else:
                    return np.array([])
            else:
                # 简化检测器（整图作为人脸）
                w, h = image.size
                return np.array([[0, 0, w, h]])
                
        except Exception as e:
            print(f"Error in detect_faces: {e}")
            return np.array([])
    
    def detect_largest_face(self, image):
        """
        检测最大的人脸
        
        Returns:
            box: [x1, y1, x2, y2] or None
        """
        boxes = self.detect_faces(image)
        
        if len(boxes) == 0:
            return None
        
        # 计算面积，返回最大的
        areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
        largest_idx = np.argmax(areas)
        
        return boxes[largest_idx]
    
    def detect_and_align(self, image):
        """
        检测并对齐人脸
        
        Returns:
            list of aligned face images
        """
        if self.mtcnn is not None:
            try:
                faces = self.mtcnn(image)
                if faces is not None:
                    if faces.dim() == 3:
                        faces = faces.unsqueeze(0)
                    return faces
            except:
                pass
        
        # Fallback: 只检测边界框
        boxes = self.detect_faces(image)
        faces = []
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = image.crop((x1, y1, x2, y2))
            face = face.resize((160, 160))
            faces.append(face)
        
        return faces