// script.js

// 0. 配置
const API_URL = 'http://127.0.0.1:8000/diagnose'; // 你的后端API地址

// --- 新增的前端多语言词典 ---
const translations = {
    en: {
        diagnoseSummary: "📊 Diagnosis Summary",
        environmentAnalysis: "🌦️ Environmental Risk Analysis",
        managementSuggestion: "📝 Management Suggestions"
    },
    ms: {
        diagnoseSummary: "📊 Ringkasan Diagnosis",
        environmentAnalysis: "🌦️ Analisis Risiko Persekitaran",
        managementSuggestion: "📝 Cadangan Pengurusan"
    },
    zh: {
        diagnoseSummary: "📊 诊断摘要",
        environmentAnalysis: "🌦️ 环境风险分析",
        managementSuggestion: "📝 管理建议"
    }
};

// 1. 获取所有需要操作的HTML元素
const imageUpload = document.getElementById('imageUpload');
const imagePreview = document.getElementById('imagePreview');
const languageSelect = document.getElementById('languageSelect');
const diagnoseBtn = document.getElementById('diagnoseBtn');
const loadingIndicator = document.getElementById('loadingIndicator');
const resultsContainer = document.getElementById('resultsContainer');

const resultTitle = document.getElementById('resultTitle');
const resultSummary = document.getElementById('resultSummary');
const resultEnvironment = document.getElementById('resultEnvironment');
const resultSuggestion = document.getElementById('resultSuggestion');

// 2. 绑定事件监听器
// 当用户选择了图片文件
imageUpload.addEventListener('change', handleImageSelect);
// 当用户点击诊断按钮
diagnoseBtn.addEventListener('click', handleDiagnoseClick);

/**
 * 当用户选择图片后，显示预览图并激活诊断按钮
 */
function handleImageSelect(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.hidden = false;
        };
        reader.readAsDataURL(file);
        diagnoseBtn.disabled = false; // 激活按钮
    }
}

/**
 * 主函数：当用户点击“进行诊断”时触发
 */
function handleDiagnoseClick() {
    const imageFile = imageUpload.files[0];
    const language = languageSelect.value;

    // 验证：确保有图片
    if (!imageFile) {
        alert("请先选择一张作物照片！");
        return;
    }

    // 更新UI：进入加载状态
    setLoadingState(true);

    // 获取GPS位置
    console.log("正在请求GPS位置...");
    if ("geolocation" in navigator) {
        navigator.geolocation.getCurrentPosition(
            (position) => {
                // 成功获取位置
                const { latitude, longitude } = position.coords;
                console.log(`GPS获取成功: Lat=${latitude}, Lon=${longitude}`);
                // 发送数据到后端
                fetchDiagnosis(imageFile, latitude, longitude, language);
            },
            (error) => {
                // 获取位置失败
                handleGeoError(error);
                setLoadingState(false); // 结束加载状态
            }
        );
    } else {
        alert("抱歉，您的浏览器不支持地理位置功能。");
        setLoadingState(false); // 结束加载状态
    }
}

/**
 * 核心：打包数据并发送到后端API
 */
async function fetchDiagnosis(image, lat, lon, lang) {
    console.log("正在构建并发送API请求...");
    const formData = new FormData();
    formData.append("image", image);
    formData.append("latitude", lat);
    formData.append("longitude", lon);
    formData.append("language", lang);

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            // 如果服务器返回了错误状态 (e.g., 400, 500)
            const errorData = await response.json();
            throw new Error(errorData.detail || `服务器错误: ${response.status}`);
        }

        const data = await response.json();
        console.log("成功接收到诊断报告:", data);
        displayResults(data);

    } catch (error) {
        console.error("API请求失败:", error);
        alert(`诊断失败: ${error.message}`);
    } finally {
        // 无论成功或失败，最后都要结束加载状态
        setLoadingState(false);
    }
}

/**
 * 将后端返回的数据，填充到HTML页面中
 */
function displayResults(data) {
    // 1. 获取用户当前选择的语言
    const selectedLang = languageSelect.value;
    // 2. 从我们的前端词典中，获取对应语言的标题文本
    const titles = translations[selectedLang];

    // 3. 更新所有小节的标题
    document.getElementById('summaryTitle').textContent = titles.diagnoseSummary;
    document.getElementById('environmentTitle').textContent = titles.environmentAnalysis;
    document.getElementById('suggestionTitle').textContent = titles.managementSuggestion;

    // 4. 更新后端返回的核心内容 (这部分和以前一样)
    resultTitle.textContent = data.title;
    resultSummary.textContent = data.diagnosis_summary;
    resultEnvironment.textContent = data.environmental_context;
    resultSuggestion.textContent = data.management_suggestion;
    
    resultsContainer.hidden = false; // 显示结果区域
}

/**
 * 控制UI的加载状态
 * @param {boolean} isLoading - 是否正在加载
 */
function setLoadingState(isLoading) {
    diagnoseBtn.disabled = isLoading;
    loadingIndicator.hidden = !isLoading;
    if (isLoading) {
        resultsContainer.hidden = true; // 加载时隐藏旧结果
    }
}

/**
 * 处理GPS获取失败的各种情况
 */
function handleGeoError(error) {
    switch (error.code) {
        case error.PERMISSION_DENIED:
            alert("您拒绝了位置权限请求。我们无法自动获取天气数据。");
            break;
        case error.POSITION_UNAVAILABLE:
            alert("无法获取当前位置信息。");
            break;
        case error.TIMEOUT:
            alert("获取位置信息超时。");
            break;
        default:
            alert("获取位置时发生未知错误。");
            break;
    }
}