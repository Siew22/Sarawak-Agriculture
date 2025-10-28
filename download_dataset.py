import os
import requests
import time
import json
from PIL import Image, ImageStat
import imagehash
from typing import List, Tuple

# --- Selenium 相关导入 ---
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# --- (狩猎清单和基础配置保持不变) ---
HUNTING_LIST = {
    '"piper nigrum" anthracnose leaf': "Pepper__Anthracnose",
    '"penyakit antraknos lada"': "Pepper__Anthracnose",
    '"胡椒 炭疽病" 叶': "Pepper__Anthracnose",
    
    '"pepper white root disease" rigidoporus': "Pepper__White_Root",
    '"penyakit akar putih lada"': "Pepper__White_Root",
    '"胡椒 白根病"': "Pepper__White_Root",
    
    '"pepper leaf blight" cercospora': "Pepper__Leaf_Blight",
    '"hawar daun lada"': "Pepper__Leaf_Blight",
    '"胡椒 叶枯病"': "Pepper__Leaf_Blight",

    '"pepper weevil damage" piper nigrum': "Pepper__Pest_Damage",
    '"kerosakan kumbang pengerat lada"': "Pepper__Pest_Damage",
    '"胡椒 象鼻虫 危害"': "Pepper__Pest_Damage",
}
IMAGE_LIMIT_PER_KEYWORD = 30
BASE_DOWNLOAD_PATH = "PlantVillage-Dataset/raw/color"

# --- 调整AI质检员的“严格程度” (已放宽) ---
MIN_RESOLUTION = (200, 200)      # 降低分辨率要求
MAX_ASPECT_RATIO_DIFF = 2.5    # 放宽长宽比
MIN_ENTROPY = 3.5              # 降低对图片复杂度的要求
DUPLICATE_HASH_THRESHOLD = 8   # 放宽对重复图片的判定


class ImageValidator:
    """我们的AI质检员，负责筛选下载的图片质量。"""
    def __init__(self):
        self.seen_hashes: List[imagehash.ImageHash] = []

    def is_valid(self, image_path: str) -> Tuple[bool, str]:
        try:
            with Image.open(image_path) as img:
                if img.width < MIN_RESOLUTION[0] or img.height < MIN_RESOLUTION[1]:
                    return False, f"分辨率太低 ({img.width}x{img.height})"
                
                if min(img.width, img.height) <= 0:
                    return False, "尺寸无效"
                aspect_ratio = max(img.width, img.height) / min(img.width, img.height)
                if aspect_ratio > MAX_ASPECT_RATIO_DIFF:
                    return False, f"长宽比异常 ({aspect_ratio:.1f})"

                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                stat = ImageStat.Stat(img)
                if sum(stat.mean) < 10:
                    return False, "图像几乎全黑"
                
                entropy = sum(stat.entropy) / len(stat.entropy)
                if entropy < MIN_ENTROPY:
                    return False, f"内容过于简单 (平均熵: {entropy:.1f})"

                current_hash = imagehash.phash(img)
                for seen_hash in self.seen_hashes:
                    if current_hash - seen_hash < DUPLICATE_HASH_THRESHOLD:
                        return False, f"与已下载图片重复 (哈希差: {current_hash - seen_hash})"
                
                self.seen_hashes.append(current_hash)
                
        except Exception as e:
            return False, f"无法打开或处理图片: {e}"
            
        return True, "合格"

def download_image(url: str, filepath: str):
    """下载单张图片，并伪装成浏览器，带详细日志"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        # --- 详细的过程监控 ---
        print(f"      - 📥 正在尝试下载: {url[:80]}...")
        response = requests.get(url, headers=headers, timeout=15, stream=True)
        response.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(8192):
                f.write(chunk)
        # --- 下载成功日志 ---
        print(f"      - ✅ 下载成功: {os.path.basename(filepath)}")
        return True
    except Exception as e:
        # --- 下载失败日志 ---
        print(f"      - ❌ 下载失败 (原因: {e})")
        return False

def get_image_urls_with_selenium(keywords: str, limit: int, driver: webdriver.Chrome) -> List[str]:
    """使用Selenium来获取Google图片搜索结果的真实URL"""
    search_url = f"https://www.google.com/search?q={requests.utils.quote(keywords)}&tbm=isch"
    driver.get(search_url)

    scroll_pause_time = 1.5
    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(3):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause_time)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
    
    image_elements = driver.find_elements(By.CSS_SELECTOR, 'img.Q4LuWd, img.YQ4gaf')
    
    urls = []
    for img in image_elements:
        if len(urls) >= limit * 2:
            break
        src = img.get_attribute('src')
        if src and src.startswith(('http', 'https')):
            urls.append(src)
    return list(set(urls))

def hunt_images():
    """主函数，协调下载和质检流程"""
    validator = ImageValidator()
    os.makedirs(BASE_DOWNLOAD_PATH, exist_ok=True)
    
    print("--- 开始自动化数据狩猎 (带详细监控) ---")

    print("正在初始化Chrome浏览器驱动...")
    try:
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument("--log-level=3") # 减少Selenium自身的日志输出
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        print("✅ 浏览器驱动初始化成功。")
    except Exception as e:
        print(f"❌ 无法初始化浏览器驱动: {e}")
        print("   请确保你的电脑上安装了Google Chrome浏览器，并且网络连接正常。")
        return

    total_downloaded = 0
    total_kept = 0
    
    for keywords, folder_name in HUNTING_LIST.items():
        validator.seen_hashes = []
        target_path = os.path.join(BASE_DOWNLOAD_PATH, folder_name)
        temp_path = os.path.join(BASE_DOWNLOAD_PATH, f"temp_{folder_name}") # 为每个类别创建独立的临时文件夹
        os.makedirs(target_path, exist_ok=True)
        os.makedirs(temp_path, exist_ok=True)
        
        print(f"\n🏹 正在狩猎: '{keywords}' (目标: {IMAGE_LIMIT_PER_KEYWORD} 张)")
        
        try:
            urls = get_image_urls_with_selenium(keywords, IMAGE_LIMIT_PER_KEYWORD, driver)
            print(f"   找到了 {len(urls)} 个潜在图片链接，开始下载和质检...")
            
            kept_count = 0
            download_count = 0
            
            for i, url in enumerate(urls):
                if kept_count >= IMAGE_LIMIT_PER_KEYWORD:
                    break

                file_extension = 'jpg'
                temp_filepath = os.path.join(temp_path, f"temp_{i}.{file_extension}")

                if download_image(url, temp_filepath):
                    download_count += 1
                    is_valid, reason = validator.is_valid(temp_filepath)
                    if is_valid:
                        final_filename = f"{folder_name}_{int(time.time() * 1000)}_{i}.jpg"
                        final_path = os.path.join(target_path, final_filename)
                        os.rename(temp_filepath, final_path)
                        kept_count += 1
                        print(f"      - ✅ 质检合格: {final_filename}")
                    else:
                        print(f"      - ❌ 质检拒绝 (原因: {reason})")
                        os.remove(temp_filepath)
                
                time.sleep(0.1)

            total_downloaded += download_count
            total_kept += kept_count
            print(f"   AI质检完成: 在尝试下载的 {download_count} 张图片中，保留了 {kept_count} 张。")

        except Exception as e:
            print(f"   ❌ 狩猎 '{keywords}' 时发生严重错误: {e}")
        finally:
            # 清理临时文件夹
            try:
                for file in os.listdir(temp_path):
                    os.remove(os.path.join(temp_path, file))
                os.rmdir(temp_path)
            except OSError:
                pass
            
    driver.quit()
    
    print("\n--- 数据狩猎完成！ ---")
    print(f"总计: 尝试下载 {total_downloaded} 张, AI质检后保留 {total_kept} 张。")

if __name__ == "__main__":
    hunt_images()