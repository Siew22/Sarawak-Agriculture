import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# --- 用户配置 ---

# 1. 定义我们所有下载的、原始的数据源
#    这是一个列表，每个元素都是一个字典，描述了数据源的信息
SOURCE_DATASETS = [
    {
        "name": "Roboflow_v1",
        "path": "Mydataset/Roboflow_v1",
        "type": "roboflow_classification" # 我们定义一种类型叫 'roboflow_classification'
    },
    {
        "name": "Roboflow_v2",
        "path": "Mydataset/roboflow_v2",
        "type": "roboflow_classification"
    },
    {
        "name": "Roboflow_v3",
        "path": "Mydataset/roboflow_v3",
        "type": "roboflow_classification"
    },
    {
        "name": "Kaggle_Pepper",
        "path": "Mydataset/kaggle_black_pepper_dataset",
        "type": "folder_per_class" # 另一种类型，每个文件夹代表一个类别
    },
    {
        "name": "Mendeley_Pepper",
        "path": "Mydataset/Pepper Diseases and Pests Detection",
        "type": "manual_sort" # 定义一种新类型，提醒自己这个需要手动处理
    },
]

# 2. 定义最终的目标“兵工厂”文件夹
MASTER_DATASET_DIR = "Masterdataset"

# 3. 定义你的类别映射
#    这里的Key，必须和原始数据集里的类别名完全一致！
#    Value 是你希望在 MasterDataset 里创建的、统一的文件夹名。
CLASS_MAPPING = {
    # 示例: 你需要根据你“侦察”到的真实类别名来填写
    # Roboflow (需要你打开 _annotations.csv 或 _classes.csv 查看)
    "black_pepper": "Pepper_Black", 
    "white_pepper": "Pepper_White",
    "Bacterial-Spot": "Pepper_Bacterial_Spot", # Roboflow 可能使用 '-'
    
    # Kaggle
    "black_pepper_healthy": "Pepper_Healthy",
    "Leaf_blight": "Pepper_Leaf_Blight",
    "Yallow_Mottle_virus": "Pepper_Yellow_Mottle_Virus",
}

def organize_roboflow_dataset(source_path, master_path, mapping):
    """处理 Roboflow 导出格式的数据集 (智能寻找标注文件并容错)"""
    print(f"\n--- 正在处理 Roboflow 数据集: {source_path} ---")
    for subset in ["train", "valid", "test"]:
        subset_path = os.path.join(source_path, subset)
        
        # --- 智能寻路逻辑 ---
        annotation_file_v1 = os.path.join(subset_path, "_annotations.csv")
        annotation_file_v2 = os.path.join(subset_path, "_classes.csv")
        
        annotation_file = None
        if os.path.exists(annotation_file_v1):
            annotation_file = annotation_file_v1
        elif os.path.exists(annotation_file_v2):
            annotation_file = annotation_file_v2
        
        # 如果找不到任何标注文件
        if not annotation_file:
            print(f"⚠️  在 '{subset_path}' 中找不到任何标注文件，将尝试作为未分类数据处理...")
            # --- 容错逻辑：处理没有标注文件的文件夹 ---
            unlabeled_target_path = os.path.join(master_path, 'train', '_unlabeled') # 全部放入训练集的'未标注'文件夹
            os.makedirs(unlabeled_target_path, exist_ok=True)
            
            if os.path.isdir(subset_path):
                image_count = 0
                for f in os.listdir(subset_path):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        # 创建一个唯一的文件名以避免冲突
                        unique_filename = f"unlabeled_{dataset_info['name']}_{subset}_{f}"
                        shutil.copy(os.path.join(subset_path, f), os.path.join(unlabeled_target_path, unique_filename))
                        image_count += 1
                if image_count > 0:
                    print(f"   已将 {image_count} 张未标注图片复制到 'train/_unlabeled' 文件夹等待手动分类。")
            continue # 处理下一个子集 (valid, test)
            
        # 如果找到了标注文件
        print(f"📂 正在使用 '{os.path.basename(annotation_file)}' 整理 '{subset}' 子集...")
        try:
            # 健壮地读取CSV，无论是否有表头
            try:
                df = pd.read_csv(annotation_file)
                if 'filename' in df.columns and 'class' in df.columns:
                     filename_col, classname_col = 'filename', 'class'
                else:
                     df = pd.read_csv(annotation_file, header=None)
                     filename_col, classname_col = 0, 1
            except Exception:
                 df = pd.read_csv(annotation_file, header=None)
                 filename_col, classname_col = 0, 1

            for _, row in df.iterrows():
                image_filename = str(row[filename_col]) # 确保是字符串
                class_name_raw = str(row[classname_col]).strip()
                
                target_folder_name = mapping.get(class_name_raw)
                if not target_folder_name: continue

                source_image_path = os.path.join(subset_path, image_filename)
                master_subset_path = os.path.join(master_path, subset, target_folder_name)
                os.makedirs(master_subset_path, exist_ok=True)
                destination_image_path = os.path.join(master_subset_path, os.path.basename(image_filename))
                
                if os.path.exists(source_image_path):
                    shutil.copy(source_image_path, destination_image_path)
        except Exception as e:
            print(f"   ❌ 处理标注文件 '{annotation_file}' 时发生错误: {e}")

def organize_folder_per_class_dataset(source_path, master_path, mapping):
    """处理每个文件夹代表一个类别的数据集 (如Kaggle)"""
    print(f"\n--- 正在处理 Folder-per-class 数据集: {source_path} ---")
    
    for class_folder_raw in os.listdir(source_path):
        target_folder_name = mapping.get(class_folder_raw)
        if not target_folder_name: continue
        
        source_class_path = os.path.join(source_path, class_folder_raw)
        if not os.path.isdir(source_class_path): continue

        print(f"📂 正在整理 '{class_folder_raw}' 类别...")
        
        images = [f for f in os.listdir(source_class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images: continue
        
        # 将这个类别的数据按 80/20 划分为训练集和验证集
        train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)
        
        # 复制训练集图片
        train_target_path = os.path.join(master_path, 'train', target_folder_name)
        os.makedirs(train_target_path, exist_ok=True)
        for img in train_images:
            shutil.copy(os.path.join(source_class_path, img), os.path.join(train_target_path, img))
            
        # 复制验证集图片
        val_target_path = os.path.join(master_path, 'valid', target_folder_name)
        os.makedirs(val_target_path, exist_ok=True)
        for img in val_images:
            shutil.copy(os.path.join(source_class_path, img), os.path.join(val_target_path, img))
            
def main():
    """主函数，协调所有数据集的整理工作"""
    if os.path.exists(MASTER_DATASET_DIR):
        print(f"警告: 目标文件夹 '{MASTER_DATASET_DIR}' 已存在。脚本会向其中添加文件。")
    os.makedirs(MASTER_DATASET_DIR, exist_ok=True)
    
    global dataset_info # 允许在 organize_roboflow_dataset 中访问
    for dataset_info in SOURCE_DATASETS:
        source_path = dataset_info["path"]
        dataset_type = dataset_info["type"]
        
        if not os.path.isdir(source_path):
            print(f"❌ 错误: 找不到源数据路径 '{source_path}'，已跳过。")
            continue
            
        if dataset_type == "roboflow_classification":
            organize_roboflow_dataset(source_path, MASTER_DATASET_DIR, CLASS_MAPPING)
        elif dataset_type == "folder_per_class":
            organize_folder_per_class_dataset(source_path, MASTER_DATASET_DIR, CLASS_MAPPING)
        elif dataset_type == "manual_sort":
            print(f"\n--- 提示: 数据集 '{dataset_info['name']}' 需要您手动分类。---")
            print(f"   请将 '{dataset_info['path']}' 里的图片，手动复制到 '{MASTER_DATASET_DIR}' 对应的 train/valid 类别文件夹中。")
        else:
            print(f"未知的数据集类型: {dataset_type}")
            
    print("\n--- ✅ 所有数据集整理完成！ ---")
    print(f"请检查 '{MASTER_DATASET_DIR}' 文件夹的结构和内容，特别是 'train/_unlabeled' 文件夹。")

if __name__ == "__main__":
    # 在运行前，请务必再次检查并修改上面的 SOURCE_DATASETS 和 CLASS_MAPPING
    main()