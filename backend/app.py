from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
import numpy as np
from PIL import Image
import io
import torch
import logging

# 导入配置
import config

# 导入自定义模块
from models.face_detector import FaceDetector
from models.pose_aware_model import PoseAwareFaceRecognition
from utils.face_database import FaceDatabase
from utils.image_processor import ImageProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=config.API_CONFIG['cors_origins'])
app.config['MAX_CONTENT_LENGTH'] = config.API_CONFIG['max_content_length']

# 初始化组件
logger.info("正在初始化系统组件...")

try:
    face_detector = FaceDetector()
    logger.info("✓ 人脸检测器加载成功")
    
    face_recognizer = PoseAwareFaceRecognition(
        trained_weights=config.TRAINED_WEIGHTS,
        hopenet_weights=config.HOPENET_WEIGHTS
    )
    logger.info("✓ 人脸识别模型加载成功")
    
    face_db = FaceDatabase(config.DATABASE_PATH)
    logger.info("✓ 人脸数据库初始化成功")
    
    image_processor = ImageProcessor()
    logger.info("✓ 图像处理器初始化成功")
    
    logger.info("=" * 60)
    logger.info("🎉 系统初始化完成！")
    logger.info(f"   数据库人脸数: {face_db.get_size()}")
    logger.info(f"   运行设备: {face_recognizer.device}")
    logger.info("=" * 60)
    
except Exception as e:
    logger.error(f"❌ 系统初始化失败: {e}")
    raise

# 配置
UPLOAD_FOLDER = config.UPLOAD_PATH
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'ok',
        'model_loaded': face_recognizer.is_loaded(),
        'database_size': face_db.get_size()
    })


@app.route('/api/enroll/single', methods=['POST'])
def enroll_single():
    """单张人脸录入"""
    try:
        data = request.json
        image_data = data.get('image')  # base64 encoded image
        name = data.get('name')
        pose = data.get('pose', 'auto')  # 新增：可选指定姿态
        
        if not image_data or not name:
            return jsonify({'error': 'Missing image or name'}), 400
        
        logger.info(f"开始录入人脸: {name}")
        
        # 解码图像
        image = image_processor.decode_base64_image(image_data)
        
        # 检测人脸
        faces = face_detector.detect_faces(image)
        
        if len(faces) == 0:
            logger.warning(f"未检测到人脸: {name}")
            return jsonify({'error': 'No face detected'}), 400
        elif len(faces) > 1:
            logger.warning(f"检测到多张人脸 ({len(faces)}): {name}")
            return jsonify({'error': f'Multiple faces detected ({len(faces)}), please upload single face image'}), 400
        
        # 提取人脸区域
        face_box = faces[0]
        face_image = image_processor.crop_face(image, face_box)
        
        # 提取特征（使用论文模型）
        logger.info(f"提取特征向量 (pose={pose})...")
        embedding = face_recognizer.extract_embedding(face_image, pose=pose)
        logger.info(f"✓ 特征向量维度: {embedding.shape}")
        
        # 保存到数据库
        face_id = face_db.add_face(name, embedding, face_image)
        
        logger.info(f"✓ 成功录入: {name} (ID: {face_id})")
        
        return jsonify({
            'success': True,
            'face_id': face_id,
            'name': name,
            'embedding_dim': len(embedding),
            'message': f'Successfully enrolled {name}'
        })
        
    except Exception as e:
        logger.error(f"录入失败: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/enroll/batch', methods=['POST'])
def enroll_batch():
    """批量录入（多张单人照片）"""
    try:
        data = request.json
        images = data.get('images')  # list of base64 encoded images
        names = data.get('names')    # list of names
        
        if not images or not names or len(images) != len(names):
            return jsonify({'error': 'Invalid images or names'}), 400
        
        results = []
        for idx, (img_data, name) in enumerate(zip(images, names)):
            try:
                image = image_processor.decode_base64_image(img_data)
                faces = face_detector.detect_faces(image)
                
                if len(faces) != 1:
                    results.append({
                        'index': idx,
                        'name': name,
                        'success': False,
                        'error': f'Detected {len(faces)} faces, expected 1'
                    })
                    continue
                
                face_box = faces[0]
                face_image = image_processor.crop_face(image, face_box)
                embedding = face_recognizer.extract_embedding(face_image)
                face_id = face_db.add_face(name, embedding, face_image)
                
                results.append({
                    'index': idx,
                    'name': name,
                    'success': True,
                    'face_id': face_id
                })
                
            except Exception as e:
                results.append({
                    'index': idx,
                    'name': name,
                    'success': False,
                    'error': str(e)
                })
        
        success_count = sum(1 for r in results if r['success'])
        
        return jsonify({
            'total': len(images),
            'success_count': success_count,
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/enroll/group', methods=['POST'])
def enroll_group():
    """从合照中批量录入"""
    try:
        data = request.json
        image_data = data.get('image')
        names = data.get('names')  # list of names for each detected face
        
        if not image_data:
            return jsonify({'error': 'Missing image'}), 400
        
        # 解码图像
        image = image_processor.decode_base64_image(image_data)
        
        # 检测所有人脸
        faces = face_detector.detect_faces(image)
        
        if len(faces) == 0:
            return jsonify({'error': 'No faces detected'}), 400
        
        # 返回检测到的人脸供用户标注
        if not names:
            face_crops = []
            for idx, face_box in enumerate(faces):
                face_image = image_processor.crop_face(image, face_box)
                face_base64 = image_processor.encode_image_to_base64(face_image)
                face_crops.append({
                    'index': idx,
                    'image': face_base64,
                    'box': face_box.tolist()
                })
            
            return jsonify({
                'detected_faces': len(faces),
                'faces': face_crops,
                'message': 'Please provide names for each face'
            })
        
        # 如果提供了名字，进行批量录入
        if len(names) != len(faces):
            return jsonify({'error': f'Names count ({len(names)}) does not match faces count ({len(faces)})'}), 400
        
        results = []
        for idx, (face_box, name) in enumerate(zip(faces, names)):
            try:
                face_image = image_processor.crop_face(image, face_box)
                embedding = face_recognizer.extract_embedding(face_image)
                face_id = face_db.add_face(name, embedding, face_image)
                
                results.append({
                    'index': idx,
                    'name': name,
                    'success': True,
                    'face_id': face_id
                })
            except Exception as e:
                results.append({
                    'index': idx,
                    'name': name,
                    'success': False,
                    'error': str(e)
                })
        
        success_count = sum(1 for r in results if r['success'])
        
        return jsonify({
            'total': len(faces),
            'success_count': success_count,
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/recognize', methods=['POST'])
def recognize():
    """人脸识别"""
    try:
        data = request.json
        image_data = data.get('image')
        threshold = data.get('threshold', 0.6)  # 相似度阈值
        
        if not image_data:
            return jsonify({'error': 'Missing image'}), 400
        
        # 解码图像
        image = image_processor.decode_base64_image(image_data)
        
        # 检测人脸
        faces = face_detector.detect_faces(image)
        
        if len(faces) == 0:
            return jsonify({
                'detected_faces': 0,
                'results': [],
                'message': 'No faces detected'
            })
        
        # 识别每个人脸
        results = []
        for idx, face_box in enumerate(faces):
            face_image = image_processor.crop_face(image, face_box)
            embedding = face_recognizer.extract_embedding(face_image)
            
            # 在数据库中搜索
            match = face_db.search_face(embedding, threshold)
            
            if match:
                results.append({
                    'index': idx,
                    'box': face_box.tolist(),
                    'recognized': True,
                    'name': match['name'],
                    'confidence': float(match['similarity']),
                    'face_id': match['face_id']
                })
            else:
                results.append({
                    'index': idx,
                    'box': face_box.tolist(),
                    'recognized': False,
                    'name': 'Unknown',
                    'confidence': 0.0
                })
        
        return jsonify({
            'detected_faces': len(faces),
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/database/list', methods=['GET'])
def list_database():
    """列出数据库中的所有人脸"""
    try:
        faces = face_db.list_all_faces()
        return jsonify({
            'total': len(faces),
            'faces': faces
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/database/delete/<face_id>', methods=['DELETE'])
def delete_face(face_id):
    """删除指定人脸"""
    try:
        success = face_db.delete_face(face_id)
        if success:
            return jsonify({'success': True, 'message': 'Face deleted successfully'})
        else:
            return jsonify({'error': 'Face not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/database/clear', methods=['DELETE'])
def clear_database():
    """清空数据库"""
    try:
        face_db.clear_all()
        return jsonify({'success': True, 'message': 'Database cleared successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 启动人脸识别服务器")
    print("=" * 60)
    
    # 验证配置
    if not config.validate_config():
        print("\n⚠️  配置验证失败，请检查上述错误！")
        print("提示: 确保以下文件存在:")
        print(f"  1. 训练权重: {config.TRAINED_WEIGHTS}")
        print(f"  2. Hopenet权重: {config.HOPENET_WEIGHTS}")
        print(f"  3. 项目源文件: Facenet_tune.py, MyModel.py等")
        exit(1)
    
    print(f"\n✓ 模型已加载: {face_recognizer.is_loaded()}")
    print(f"✓ 数据库大小: {face_db.get_size()}")
    print(f"✓ 服务地址: http://{config.API_CONFIG['host']}:{config.API_CONFIG['port']}")
    print("=" * 60 + "\n")
    
    app.run(
        host=config.API_CONFIG['host'],
        port=config.API_CONFIG['port'],
        debug=config.API_CONFIG['debug']
    )