# app/models/disease_classifier.py (最终自研加载与深度诊断版)
import torch
import torch.nn as nn # 导入 nn 模块
import json
from pathlib import Path
from ..schemas.diagnosis import PredictionResult
from typing import Dict, Tuple

# 关键：从 torchvision.models 导入的是模型结构本身
from torchvision.models import efficientnet_b0, efficientnet_b2 

class DiseaseClassifier:
    def __init__(self, model_path: Path, labels_path: Path, architecture: str = 'b0'):
        """
        初始化分类器。
        :param model_path: 你自己训练的模型权重文件 (.pth) 的路径。
        :param labels_path: 标签映射文件 (.json) 的路径。
        :param architecture: 'b0' 或 'b2'，必须与你训练时使用的模型架构一致！
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DiseaseClassifier正在使用设备: {self.device}")

        try:
            # 1. 加载标签文件，确定类别数
            with open(labels_path) as f:
                self.labels = {int(k): v for k, v in json.load(f).items()}
            num_classes = len(self.labels)

            # 2. 构建一个“空白”的、与你训练时完全相同的模型结构
            print(f"正在构建一个全新的 '{architecture}' 模型结构 (用于加载自研权重)...")
            if architecture == 'b0':
                # --- 核心修改：weights=None 确保不加载任何预训练权重 ---
                self.model = efficientnet_b0(weights=None, num_classes=num_classes)
            elif architecture == 'b2':
                # --- 核心修改：weights=None 确保不加载任何预训练权重 ---
                self.model = efficientnet_b2(weights=None, num_classes=num_classes)
            else:
                raise ValueError(f"不支持的模型架构: {architecture}。请选择 'b0' 或 'b2'。")

            # 3. 将你亲手训练的权重，加载到这个“空白”的结构中
            print(f"正在加载你的自研模型权重从: {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            
            self.model.to(self.device)
            self.model.eval()  # 设定为评估模式

        except FileNotFoundError:
            raise RuntimeError(f"模型文件或标签文件未找到。请确保 '{model_path}' 和 '{labels_path}' 存在。")
        except Exception as e:
            raise RuntimeError(f"加载自研模型时出错: {e}")

    def predict(self, image_tensor) -> Tuple[PredictionResult, Dict[str, str]]:
        """
        升级版：返回最高概率的预测结果，以及所有类别的前5名概率字典
        """
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # 获取最高概率的预测结果
            confidence_tensor, predicted_idx_tensor = torch.max(probabilities, 0)
            top_prediction = PredictionResult(
                disease=self.labels.get(predicted_idx_tensor.item(), "未知病害"),
                confidence=confidence_tensor.item()
            )

            # 获取概率最高的前5名
            top5_prob, top5_indices = torch.topk(probabilities, 5)
            
            top5_results = {}
            for i in range(top5_prob.size(0)):
                idx = top5_indices[i].item()
                label = self.labels.get(idx, f"未知类别_{idx}")
                prob = top5_prob[i].item()
                top5_results[label] = f"{prob:.2%}"

            return top_prediction, top5_results

# --- 全局实例初始化 ---
# 在这里，你需要明确指定你现在要加载的是哪个模型！
# 假设你现在使用的是6GB VRAM训练出的B0模型
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# 在这里，你需要明确指定你现在要加载的是哪个模型！
# 假设你现在使用的是6GB VRAM训练出的B0模型
#MODEL_PATH = BASE_DIR / "models_store" / "scratch_model_b0_6gb.pth"
#LABELS_PATH = BASE_DIR / "models_store" / "disease_labels.json"
#MODEL_ARCH = "b0" # 明确告诉分类器，我们用的是b0架构

# 如果你要使用12GB VRAM训练出的B2模型，就把上面三行改成：
MODEL_PATH = BASE_DIR / "models_store" / "true_scratch_model_b2_v1.pth"
LABELS_PATH = BASE_DIR / "models_store" / "disease_labels.json"
MODEL_ARCH = "b2"

# 创建一个全局分类器实例
classifier = DiseaseClassifier(model_path=MODEL_PATH, labels_path=LABELS_PATH, architecture=MODEL_ARCH)

# 创建一个全局分类器实例
classifier = DiseaseClassifier(model_path=MODEL_PATH, labels_path=LABELS_PATH, architecture=MODEL_ARCH)