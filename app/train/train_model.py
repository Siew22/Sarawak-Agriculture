# train/train_model.py (最终的、绝对正确的6GB经济版)
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
import os
import json
from tqdm import tqdm
import time
from PIL import Image

# --- 6GB VRAM 自研配置 (多数据源) ---
DATA_DIRS = [
    '../../PlantVillage-Dataset/raw/color', 
    '../../Mydataset/kaggle_black_pepper_dataset',
    '../../Mydataset/Pepper Diseases and Pests Detection/Pepper Diseases and Pests Detection/PepperDiseaseTest',
    '../../PlantVillage-Dataset/raw/grayscale',
    '../../PlantVillage-Dataset/raw/segmented',
    #'../../PlantVillage-Dataset/data_distribution_for_SVM/train',
    #'../../PlantVillage-Dataset/data_distribution_for_SVM/test',
]
MODEL_SAVE_PATH = '../../models_store/PEPPER_ONLY_model_b0_FINAL.pth'
LABELS_PATH = '../../models_store/pepper_only_labels.json'
BATCH_SIZE = 16 # 保持6GB安全值
NUM_WORKERS = 2 # 保持6GB安全值
NUM_EPOCHS = 60
LEARNING_RATE = 0.001
MODEL_ARCHITECTURE = 'efficientnet_b0'


# --- 关键修复：将所有类都定义在函数外部！ ---
class FinalPepperDataset(Dataset):
    """
    一个终极的自定义数据集类。
    它直接接收一个包含 (图片路径, 标签索引) 的样本列表来工作。
    """
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"\n警告: 加载图片 {path} 时出错: {e}. 将尝试加载下一个样本。")
            if len(self) == 0: return None, None
            return self.__getitem__((idx + 1) % len(self))

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu': 
        print("⚠️  警告: 未检测到CUDA, 训练会非常慢。")
    else: 
        print(f"✅ 检测到CUDA设备: {torch.cuda.get_device_name(0)}")

    # 1. “胡椒专属”类别扫描
    print("正在扫描所有数据源并智能筛选'Pepper'相关类别...")
    pepper_class_names = set()
    for data_dir in DATA_DIRS:
        if os.path.isdir(data_dir):
            for class_name in os.listdir(data_dir):
                if os.path.isdir(os.path.join(data_dir, class_name)) and 'pepper' in class_name.lower():
                    pepper_class_names.add(class_name)
    
    sorted_class_names = sorted(list(pepper_class_names))
    if not sorted_class_names:
        print("❌ 错误：在所有数据源中，没有找到任何包含'pepper'字眼的类别文件夹！")
        return
        
    NUM_CLASSES = len(sorted_class_names)
    idx_to_class = {str(i): name for i, name in enumerate(sorted_class_names)}
    class_to_idx = {name: i for i, name in enumerate(sorted_class_names)}
    
    os.makedirs(os.path.dirname(LABELS_PATH), exist_ok=True)
    with open(LABELS_PATH, 'w') as f: json.dump(idx_to_class, f, indent=4)
    print(f"✅ 标签文件已更新，共找到 {NUM_CLASSES} 个与'Pepper'相关的唯一类别。")
    print("将要训练的类别:", sorted_class_names)

    # 2. 回归稳定可靠的 torchvision.transforms (6GB优化版)
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }

    # 3. 最终的、绝对正确的数据加载逻辑
    print("\n正在从多个数据源中加载并合并'Pepper'相关的数据集...")
    all_pepper_samples = []

    for data_dir in DATA_DIRS:
        if not os.path.isdir(data_dir):
            continue
        
        print(f"  - 正在扫描: {data_dir}")
        for class_name in os.listdir(data_dir):
            if class_name in pepper_class_names:
                class_path = os.path.join(data_dir, class_name)
                if os.path.isdir(class_path):
                    image_count = 0
                    for img_file in os.listdir(class_path):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            all_pepper_samples.append((os.path.join(class_path, img_file), class_name))
                            image_count += 1
                    if image_count > 0:
                        print(f"    - ✅ 找到了 '{class_name}' 类别，并添加了 {image_count} 张图片。")
    
    if not all_pepper_samples:
        print("❌ 错误: 在所有数据源中，没有找到任何有效的'Pepper'图片！")
        return

    final_samples = [(path, class_to_idx[label]) for path, label in all_pepper_samples]
    
    full_dataset = FinalPepperDataset(final_samples)
    print(f"\n✅ 数据整合完成: 共 {len(full_dataset)} 张'Pepper'相关图片。")

    print("正在将总数据集划分为训练集和验证集 (80/20)...")
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_split, val_split = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=generator)
    
    # 为划分后的子集应用不同的transform
    train_split.dataset.transform = data_transforms['train']
    val_split.dataset.transform = data_transforms['val']
    
    dataloaders = {
        'train': DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True),
        'val': DataLoader(val_split, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True),
    }
    print(f"✅ 数据加载器准备就绪: 训练集 {len(train_split)} 张, 验证集 {len(val_split)} 张。")

    print(f"正在构建一个全新的 '{MODEL_ARCHITECTURE}' 模型 (100% 完全自研)...")
    model = models.efficientnet_b0(weights=None, num_classes=NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    start_time = time.time()
    best_acc = 0.0

    print("\n--- 开始“胡椒专属”优化训练 (6GB版) ---")
    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS} | 当前学习率: {optimizer.param_groups[0]["lr"]:.6f}')
        print('-' * 25)
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects = 0.0, 0
            
            progress_bar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch+1}")
            for inputs, labels in progress_bar:
                if inputs is None: continue
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                progress_bar.set_postfix(loss=f'{loss.item():.4f}')

            if len(dataloaders[phase].dataset) > 0:
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
                    print(f"🎉 新的最佳自研经济版模型已保存 (Accuracy: {best_acc:.4f}) 🎉")
        
        scheduler.step()

    time_elapsed = time.time() - start_time
    print(f'\n--- 训练完成 ---')
    print(f'总耗时: {time_elapsed // 60:.0f}分 {time_elapsed % 60:.0f}秒')
    print(f'🏆 最佳验证集准确率: {best_acc:4f}')

if __name__ == "__main__":
    train()