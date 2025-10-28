from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware # 1. 导入CORS中间件
from fastapi.responses import JSONResponse
#import logging
from loguru import logger
from typing import Dict, Any

# 导入我们自己创建的所有模块和服务
from .schemas.diagnosis import FullDiagnosisReport
from .utils.image_processing import image_processor
from .models.disease_classifier import classifier
from .models.risk_assessor import risk_assessor
from .models.recommendation_generator import report_generator_v2 as report_generator
from .services.weather_service import weather_service
from loguru import logger
import logging
# 导入NLP服务 (暂时未在核心端点使用，为未来预留)
# from .services.nlp_service import nlp_service 

# --- 初始化与配置 ---

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建FastAPI应用实例
app = FastAPI(
    title="Sarawak Agri-Advisor API",
    description="一个基于AI和软计算的智能农业诊断系统，旨在赋能砂拉越的农户。版本2实现了自动化天气获取和动态报告生成。",
    version="2.0.0"
)

# --- 2. 添加CORS中间件配置 (关键步骤) ---
# 这个配置必须放在所有API路由定义之前

# 定义允许访问此API的来源列表。
# 在开发阶段，使用 "*" 是最简单的方法，它允许来自任何地方的请求。
# 在生产环境中，为了安全，应该将其替换为你的前端应用的实际域名。
origins = [
    "*",  # 允许所有来源
    # 例如，未来部署后可能会是:
    # "http://sarawak-agri-advisor.com",
    # "https://sarawak-agri-advisor.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,  # 允许cookies (虽然我们目前没用，但加上是好习惯)
    allow_methods=["*"],     # 允许所有HTTP方法 (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],     # 允许所有HTTP请求头
)


# --- API生命周期事件 ---

@app.on_event("startup")
async def startup_event():
    """在应用启动时执行的异步函数"""
    logger.info("Sarawak Agri-Advisor API 正在启动...")
    logger.info(f"AI模型已在设备 {classifier.device} 上准备就绪。")
    logger.info("所有服务和模块已初始化。")
    logger.info("CORS策略已启用，允许所有来源。")
    logger.info("应用启动完成，等待请求...")

@app.on_event("shutdown")
async def shutdown_event():
    """在应用关闭时执行的异步函数"""
    logger.info("Sarawak Agri-Advisor API 正在关闭...")


# --- 依赖注入 (Dependency Injection) ---

async def get_weather_data(
    latitude: float = Form(..., description="用户设备的GPS纬度", ge=-90, le=90),
    longitude: float = Form(..., description="用户设备的GPS经度", ge=-180, le=180)
) -> Dict[str, Any]:
    """一个可重用的依赖项，用于从前端提供的经纬度自动获取天气数据。"""
    try:
        weather_data = await weather_service.get_current_weather(latitude, longitude)
        logger.info(f"成功获取天气数据: Temp={weather_data['temperature']}°C, Humidity={weather_data['humidity']}%")
        return weather_data
    except ConnectionError as e:
        logger.error(f"无法连接到天气服务: {e}")
        raise HTTPException(status_code=503, detail=f"天气服务当前不可用，请稍后再试。")


# --- API 端点 (API Endpoints) ---

@app.get("/", summary="API 健康检查", tags=["General"])
def read_root():
    """一个简单的端点，用于检查API服务是否正在运行。"""
    return {"status": "ok", "message": "欢迎使用 Sarawak Agri-Advisor API V2！"}

@app.post("/diagnose", response_model=FullDiagnosisReport, summary="获取完整的作物健康诊断报告", tags=["Diagnosis"])
async def create_diagnosis_report(
    image: UploadFile = File(..., description="待诊断的作物叶片图片"),
    weather: Dict[str, Any] = Depends(get_weather_data), # 使用依赖注入获取天气数据
    language: str = Form("en", description="期望的报告语言 (en, ms, zh)", enum=["en", "ms", "zh"])
):
    """
    这是我们的核心功能端点。用户上传一张作物图片和GPS坐标，系统将自动完成所有分析并返回一份完整的诊断报告。
    """
    try:
        # 1. 提取已自动获取的天气数据
        temperature = weather["temperature"]
        humidity = weather["humidity"]
        
        # 2. 读取并验证图片文件
        logger.info(f"接收到文件: {image.filename}, 内容类型: {image.content_type}")
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="上传的文件必须是图片格式。")
        
        image_bytes = await image.read()
        
        # 3. 图像预处理
        image_tensor = image_processor.process(image_bytes)
        
        # AI模型进行病害预测
        top_prediction, top5_probabilities = classifier.predict(image_tensor)

        # --- ↓↓↓ 新增的“安全阀”逻辑 ↓↓↓ ---
        CONFIDENCE_THRESHOLD = 0.75 # 设定一个75%的置信度门槛

        if top_prediction.confidence < CONFIDENCE_THRESHOLD:
            # 如果AI的最高自信度都低于75%，我们不采纳它的答案
            # 而是强制生成一个“无法确定”的特殊报告
            logger.warning(f"模型置信度 ({top_prediction.confidence:.2%}) 低于阈值 {CONFIDENCE_THRESHOLD:.0%}")
            # 我们伪造一个 'unknown' 的预测结果
            uncertain_prediction = PredictionResult(disease="unknown", confidence=top_prediction.confidence)
            report = report_generator.generate(uncertain_prediction, risk, lang=language)
            return report
        # --- 安全阀结束 ---

        # ... (如果置信度足够高，则继续正常流程) ...
        logger.info(f"模型最高预测: {top_prediction}")
        # ... (日志、风险评估、报告生成等) ...
        
        report = report_generator.generate(top_prediction, risk, lang=language)
        return report

    except ValueError as e:
        logger.error(f"数据处理错误: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理诊断请求时发生未知服务器错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="服务器内部发生意外错误，我们的团队已收到通知。")