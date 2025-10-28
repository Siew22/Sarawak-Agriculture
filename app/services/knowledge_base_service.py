# app/services/knowledge_base_service.py
import yaml
from pathlib import Path
from typing import Dict, Any

class KnowledgeBaseService:
    def __init__(self, kb_path: Path):
        self.knowledge_base = self._load_all_knowledge(kb_path)
        print(f"知识库加载成功，共加载 {len(self.knowledge_base)} 条病害记录。")

    def _load_all_knowledge(self, kb_path: Path) -> Dict[str, Any]:
        """加载 knowledge_base 文件夹下所有的 .yaml 文件"""
        full_kb = {}
        if not kb_path.is_dir():
            print(f"警告: 知识库路径 '{kb_path}' 不存在。")
            return full_kb
            
        for yaml_file in kb_path.glob("*.yaml"):
            with open(yaml_file, 'r', encoding='utf-8') as f:
                try:
                    data = yaml.safe_load(f)
                    if data:
                        # --- ↓↓↓ 在这里进行修复 ↓↓↓ ---
                        for key, value in data.items():
                            # 将key进行标准化，去除可能存在的多余空格或特殊字符
                            cleaned_key = str(key).strip()
                            full_kb[cleaned_key] = value
                        # --- ↑↑↑ 修复结束 ↑↑↑ ---
                except yaml.YAMLError as e:
                    print(f"加载YAML文件 '{yaml_file.name}' 时出错: {e}")
        return full_kb

    def get_disease_info(self, disease_key: str) -> Dict[str, Any]:
        """根据内部病害名称 (key) 获取知识"""
        return self.knowledge_base.get(disease_key)

# --- 全局实例初始化 ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
KB_PATH = BASE_DIR / "knowledge_base"

# 创建一个全局知识库服务实例
kb_service = KnowledgeBaseService(kb_path=KB_PATH)