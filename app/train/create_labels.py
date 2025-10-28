# train/create_labels.py (已修复路径问题)
import os
import json

# --- 核心修改点：调整相对路径 ---
# 从 train -> app -> sarawak_agri (根目录)
# 所以我们需要使用两次 '../'
DATA_DIR = '../../PlantVillage-Dataset/raw/color'
OUTPUT_PATH = '../../models_store/disease_labels.json'

def create_label_mapping():
    # 确保目标文件夹存在
    output_dir = os.path.dirname(OUTPUT_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建了文件夹: {output_dir}")
    
    # 检查数据源是否存在
    if not os.path.isdir(DATA_DIR):
        print(f"错误: 找不到数据集路径 '{DATA_DIR}'")
        print("请确保你已经将 PlantVillage-Dataset 下载并放置在项目根目录下。")
        return

    # 获取所有子文件夹（即类别）的名称
    class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    
    # 创建从索引到类名的映射 (确保key是字符串，以符合JSON标准)
    label_map = {str(i): class_name for i, class_name in enumerate(class_names)}
    
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(label_map, f, indent=4)
    
    print(f"标签映射文件已成功创建于: {OUTPUT_PATH}")
    print(f"总共找到 {len(class_names)} 个类别。")

if __name__ == "__main__":
    create_label_mapping()