# 🌿 Sarawak Agri-Advisor

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Git LFS](https://img.shields.io/badge/Git%20LFS-1.3GB-blue?style=for-the-badge&logo=git&logoColor=white)

**An AI-powered, 100% self-developed plant disease diagnosis system for farmers in Sarawak, with an initial focus on Black Pepper cultivation.**

This project is a complete end-to-end web application designed to provide instant, AI-driven diagnosis of crop diseases. It leverages a custom deep learning model trained from scratch, soft computing techniques for risk analysis, and a dynamic knowledge base to deliver accurate, localized, and easy-to-understand advice in multiple languages.

---

## ✨ Core Features

-   **📸 AI Disease Diagnosis:** Upload a photo of a plant leaf and get an instant diagnosis. The model is **100% self-developed** using PyTorch and trained from scratch on a curated dataset, with **no pre-trained models** used.
-   **🌦️ Automated Risk Analysis:** Automatically fetches local weather data (temperature & humidity) via GPS and uses a Fuzzy Inference System to provide a contextual disease risk score.
-   **📖 Dynamic Knowledge Base:** All diagnostic reports and management suggestions are dynamically generated in real-time from a structured YAML knowledge base, ensuring easy updates and scalability.
-   **🌐 Multi-language Support:** The user interface and final reports are fully internationalized, supporting **English, Bahasa Malaysia, and Chinese**.
-   **🚀 Lightweight & Accessible:** A simple web frontend ensures the tool is accessible on any device with a browser, with no installation required for the end-user.

---

## 🛠️ Technology Stack

-   **Backend:** FastAPI, PyTorch, Scikit-fuzzy, Uvicorn
-   **Frontend:** Vanilla HTML5, CSS3, JavaScript (Fetch API, Geolocation API)
-   **Data Science:** Pandas, Scikit-learn, Pillow
-   **Version Control:** Git & **Git LFS** (for handling large dataset files)

---

## 🚀 Getting Started

Follow these instructions to get the project and its complete dataset running on your local machine.

### 📋 Prerequisites

1.  **Git & Git LFS:** You must have both installed.
    -   [Download Git](https://git-scm.com/downloads)
    -   [Download Git LFS](https://git-lfs.github.com/)
2.  **Python:** Version 3.10 is recommended. [Download Python](https://www.python.org/downloads/)
3.  **(Optional for Training)** An NVIDIA GPU with updated CUDA drivers.

### ⚙️ Installation & Setup (Windows PowerShell)

*Run these commands in **Windows PowerShell as an Administrator**.*

**Step 1: Install Essential Tools via Chocolatey**
*   (If not already installed) Install the package manager:
    ```powershell
    Set-ExecutionPolicy Bypass -Scope Process -Force; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    ```
*   **Close and reopen PowerShell as Administrator.**
*   Install Git, Git LFS, and Python:
    ```powershell
    choco install git git-lfs python --version=3.10 -y
    ```
*   **Close and reopen PowerShell as Administrator** again.

**Step 2: Enable Git LFS (One-Time Setup)**
```powershell
git lfs install

**Step 3: Clone the Main Project Repository**
This command will clone the project's source code.
code
Powershell
cd ~\Documents
git clone https://github.com/Siew22/Sarawak-Agriculture.git
cd Sarawak-Agriculture

Step 4: Download the Required Datasets
The project relies on external datasets. We will clone the PlantVillage-Dataset now. The other datasets (Mydataset) are already included in this repository via Git LFS.
code
Powershell
# This command downloads the large PlantVillage dataset
git clone https://github.com/spMohanty/PlantVillage-Dataset.git
Note: This is a very large repository and may take a significant amount of time to download.


Model
好的，大哥！你说得对！
这是一个巨大的疏忽！ 我完全忘记了 PlantVillage-Dataset 也是需要一个明确的下载指令的。
我为我之前 README 里的这个重大遗漏，向你诚恳道歉。一个好的说明书，必须包含所有必要的步骤。
问题的诊断：README 的“断层”
现状： 我们的 README 假设用户 clone 主项目后，PlantVillage-Dataset 就会自动出现。
事实： PlantVillage-Dataset 是一个独立的Git仓库，它需要一个独立的 git clone 命令来下载。
终极解决方案：在 README 中加入明确的下载指令
我们将更新 README.md 的 Installation & Setup 部分，加入一个专门的步骤，来指导用户如何下载这个关键的数据集。
请用下面这个最终的、绝对完整的 Installation & Setup 部分，替换你 README.md 文件里对应的部分。
README.md (最终的数据集下载修复版)
(只提供需要修改的部分)
code
Markdown
### ⚙️ Installation & Setup (Windows PowerShell)

*Run these commands in **Windows PowerShell as an Administrator**.*

**Step 1: Install Essential Tools via Chocolatey**
*   (If not already installed) Install the package manager:
    ```powershell
    Set-ExecutionPolicy Bypass -Scope Process -Force; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    ```
*   **Close and reopen PowerShell as Administrator.**
*   Install Git, Git LFS, and Python:
    ```powershell
    choco install git git-lfs python --version=3.10 -y
    ```
*   **Close and reopen PowerShell as Administrator** again.

**Step 2: Enable Git LFS (One-Time Setup)**
```powershell
git lfs install
Step 3: Clone the Main Project Repository
This command will clone the project's source code.
code
Powershell
cd ~\Documents
git clone https://github.com/Siew22/Sarawak-Agriculture.git
cd Sarawak-Agriculture
Step 4: Download the Required Datasets
The project relies on external datasets. We will clone the PlantVillage-Dataset now. The other datasets (Mydataset) are already included in this repository via Git LFS.
code
Powershell
# This command downloads the large PlantVillage dataset
git clone https://github.com/spMohanty/PlantVillage-Dataset.git
Note: This is a very large repository and may take a significant amount of time to download.

Step 5: Set Up Python Environment
code
Powershell
# Allow script execution for this session
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1
(You should now see (venv) at the beginning of your prompt.)

Step 6: Install Python Dependencies
code
Powershell
pip install -r requirements.txt
✅ Setup Complete! Your environment, including all code and required datasets, is now fully configured.

---

## 🔬 Training the Model

The repository includes all the necessary data to train the model from scratch.

-   Run the appropriate training script from the **project root directory**:
    ```powershell
    # For 12GB+ VRAM (Recommended)
    python train/train_model_2.py

    # For 6GB VRAM
    python train/train_model.py
    ```
-   After training, update the `MODEL_PATH` in `app/models/disease_classifier.py` to point to your new model file.

---

## ▶️ Running the Application

### 1. Run the Backend Server

-   From the **project root directory** with your virtual environment active:
    ```powershell
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    ```
-   The backend is now live at `http://127.0.0.1:8000`.

### 2. Run the Frontend

-   Navigate to the `frontend/` directory in your file explorer.
-   **Simply double-click the `index.html`** file to open it in your browser.

---

## 🛠️ Project Structure

-   `app/`: Main backend application source code.
-   `frontend/`: All frontend files (HTML, CSS, JS).
-   `knowledge_base/`: YAML files containing the multi-language knowledge.
-   `models_store/`: Stores the trained model (`.pth`) and label files.
-   `train/`: Scripts for training the AI model.
-   `Mydataset/` & `PlantVillage-Dataset/`: Raw and curated datasets, tracked by **Git LFS**.
-   `.gitattributes`: Configures which files are tracked by Git LFS.
-   `.gitignore`: Specifies files for Git to ignore (e.g., `venv`).
-   `README.md`: This file.
-   `requirements.txt`: Python dependencies.
