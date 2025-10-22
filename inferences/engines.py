import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import toml
import numpy as np
from PIL import Image
from models import AnomalyDetectionModel

# 加载配置
configs = toml.load('D:\Python\Surveillance-main-v1\inferences\configs\config.toml')

# 设备配置
device = torch.device(configs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

# 视频处理参数
width = configs['segment-width']
height = configs['segment-height']
length = configs['segment-length']

x1 = configs['crop-x1']
x2 = configs['crop-x2']
y1 = configs['crop-y1']
y2 = configs['crop-y2']

# 平滑参数
smoothing_weight = np.ones(configs['smoothing-window']) / configs['smoothing-window']


class FeatureExtractor:
    """特征提取器 - 使用ResNet50（与sequential_frame_preprocessor.py完全兼容）"""

    def __init__(self, target_feature_dim=2304, adapter_path=None):
        self.device = device
        self.target_feature_dim = target_feature_dim

        # 加载ResNet50模型
        self.model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.to(self.device)
        self.model.eval()

        # 特征维度适配器
        self.feature_adapter = None
        if adapter_path and os.path.exists(adapter_path):
            # 加载预处理时保存的adapter权重
            self.feature_adapter = nn.Linear(2048, target_feature_dim).to(self.device)
            self.feature_adapter.load_state_dict(torch.load(adapter_path, map_location=self.device))
            self.feature_adapter.eval()
            print(f"已加载特征适配器: {adapter_path}")
        else:
            # 创建新的adapter（需要与预处理时保持一致）
            self.feature_adapter = nn.Linear(2048, target_feature_dim).to(self.device)
            self.feature_adapter.eval()
            print("警告: 使用随机初始化的特征适配器，可能与训练数据不一致！")
            print("建议: 在预处理时保存adapter权重，推理时加载相同的adapter")

        # 图像预处理（与sequential_frame_preprocessor.py完全一致）
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def extract_frame_feature(self, frame):
        """提取单帧特征"""
        # 转换为PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # 预处理
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 特征提取
        with torch.no_grad():
            features = self.model(image_tensor)
            features = features.flatten()

            # 调整特征维度到目标维度
            if features.shape[0] != self.target_feature_dim:
                if self.feature_adapter is None:
                    self.feature_adapter = nn.Linear(
                        features.shape[0], self.target_feature_dim
                    ).to(self.device)
                features = self.feature_adapter(features)

        return features.cpu().numpy()


class AnomalyDetector:
    """异常检测器"""

    def __init__(self, model_path, attention_window=5, alpha=0.5):
        self.device = device

        # 加载检测模型
        self.model = AnomalyDetectionModel(
            attention_window=attention_window,
            alpha=alpha
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)
        self.model.eval()

    def detect(self, features):
        """检测异常"""
        # 转换为tensor
        features_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(features_tensor)
            scores = torch.sigmoid(outputs).squeeze(0)

        return scores.cpu().numpy()


# 初始化提取器和检测器
feature_extractor = FeatureExtractor(
    target_feature_dim=2304,
    adapter_path=configs.get('feature-adapter-path')  # 可选：加载预处理时的adapter
)
anomaly_detector = AnomalyDetector(
    model_path=configs['detection-model-path'],
    attention_window=configs.get('attention-window', 5),
    alpha=configs.get('alpha', 0.5)
)


def frame_preprocess(frame):
    """帧预处理"""
    preprocessed = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    preprocessed = preprocessed[y1:y2, x1:x2]
    return preprocessed


def load_next_segment(capture):
    """加载下一个视频段"""
    segment_frames = []

    while len(segment_frames) < length:
        read_success, captured_frame = capture.read()

        if read_success:
            segment_frames.append(frame_preprocess(captured_frame))
        else:
            return False, None

    return True, segment_frames


def extract_segment_features(frames):
    """提取视频段特征"""
    features = []
    for frame in frames:
        feature = feature_extractor.extract_frame_feature(frame)
        features.append(feature)

    return np.stack(features, axis=0)


def extract_video_features(video_path):
    """提取视频特征"""
    features = []
    capture = cv2.VideoCapture(video_path)

    while capture.isOpened():
        load_success, frames = load_next_segment(capture)

        if load_success:
            segment_features = extract_segment_features(frames)
            features.append(segment_features)
        else:
            capture.release()
            break

    if len(features) == 0:
        return None

    # 拼接所有段的特征
    return np.concatenate(features, axis=0)


def detection_by_features(features):
    """基于特征进行检测"""
    return anomaly_detector.detect(features)


def score_smoothing(scores):
    """分数平滑"""
    return np.convolve(scores, smoothing_weight, mode='same').round(decimals=2)


def expand_scores(scores):
    """扩展分数（如果需要逐帧分数）"""
    return np.array(scores).repeat(length, axis=0)


def detection_by_video(video_path):
    """基于视频进行检测"""
    features = extract_video_features(video_path)

    if features is None:
        return np.array([])

    scores = detection_by_features(features)
    return score_smoothing(scores)


def anomaly_prompt_enhancement(frame, prompt):
    """添加异常提示文字"""
    return cv2.putText(frame, prompt, (120, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 215), thickness=2)


def anomaly_border_enhancement(frame, border):
    """添加异常边框"""
    x2 = frame.shape[1]
    y2 = frame.shape[0]
    return cv2.rectangle(frame, (0, 0), (x2, y2), (0, 0, 215), thickness=border)


def draw_detection_result(frame, score):
    """绘制检测结果"""
    if frame is None:
        raise ValueError("输入帧为 None")

    # 创建副本避免修改原帧
    result = frame.copy()

    try:
        if score > configs['anomaly-threshold']:
            result = anomaly_prompt_enhancement(result, configs['anomaly-prompt'])
            result = anomaly_border_enhancement(result, configs['anomaly-border'])
            result = cv2.putText(result, f'{score:.2f}', (30, 60),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 215), thickness=2)
        else:
            result = cv2.putText(result, f'{score:.2f}', (30, 60),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 215, 0), thickness=2)
    except Exception as e:
        print(f"⚠ 绘制检测结果时出错: {e}")
        # 返回原帧
        return frame

    return result