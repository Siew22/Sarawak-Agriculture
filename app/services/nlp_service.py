# app/services/nlp_service.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from typing import List, Dict

class NlpService:
    def __init__(self):
        # 1. 构建我们的多语言训练数据集 (内存中)
        # 未来可以从YAML或CSV文件加载
        training_data: List[Dict[str, str]] = [
            # --- Leaf Spot Symptoms ---
            {"text": "my pepper leaves have black spots", "label": "leaf_spot"},
            {"text": "dark spots on the leaf", "label": "leaf_spot"},
            {"text": "daun lada saya ada bintik hitam", "label": "leaf_spot"},
            {"text": "bintik gelap atas daun", "label": "leaf_spot"},
            {"text": "我的辣椒叶子有黑点", "label": "leaf_spot"},
            {"text": "叶片上有深色斑点", "label": "leaf_spot"},
            
            # --- Yellowing Symptoms ---
            {"text": "the leaves are turning yellow", "label": "yellowing"},
            {"text": "why is my plant yellow", "label": "yellowing"},
            {"text": "daun menjadi kuning", "label": "yellowing"},
            {"text": "daun nampak tidak sihat dan kuning", "label": "yellowing"},
            {"text": "叶子正在变黄", "label": "yellowing"},
            {"text": "植株发黄是什么问题", "label": "yellowing"},

            # --- Wilting Symptoms ---
            {"text": "my plant is wilting", "label": "wilting"},
            {"text": "pokok saya layu", "label": "wilting"},
            {"text": "我的植物萎蔫了", "label": "wilting"}
        ]

        X_train = [item["text"] for item in training_data]
        y_train = [item["label"] for item in training_data]

        # 2. 构建一个简单的NLP处理管道
        # TfidfVectorizer: 将文本转换成数字向量
        # LinearSVC: 一个快速且高效的分类器 (支持向量机)
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2))), # 考虑单个词和两个词的组合
            ('clf', LinearSVC()),
        ])

        # 3. "训练"我们的管道
        print("正在训练轻量级NLP意图识别模型...")
        self.pipeline.fit(X_train, y_train)
        print("NLP模型训练完成。")

    def detect_symptoms(self, text: str) -> List[str]:
        """从用户输入的文本中识别出症状标签"""
        if not text.strip():
            return []
            
        # 使用我们训练好的管道进行预测
        # 我们将输入文本放在一个列表中，因为.predict期望一个可迭代对象
        predictions = self.pipeline.predict([text])
        
        # .predict返回一个numpy数组，我们取第一个元素
        return [predictions[0]] if predictions else []

# 创建一个全局实例
nlp_service = NlpService()