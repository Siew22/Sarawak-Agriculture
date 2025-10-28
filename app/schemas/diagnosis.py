from pydantic import BaseModel, Field
from typing import Optional

class PredictionResult(BaseModel):
    disease: str = Field(..., description="预测出的病害名称")
    confidence: float = Field(..., description="模型的置信度分数 (0.0 to 1.0)")

class RiskAssessment(BaseModel):
    risk_score: float = Field(..., description="环境风险评分 (0 to 10)")
    risk_level: str = Field(..., description="风险等级 (Low, Medium, High)")

class FullDiagnosisReport(BaseModel):
    title: str = Field(..., description="报告标题")
    diagnosis_summary: str = Field(..., description="诊断结果摘要")
    environmental_context: str = Field(..., description="结合环境因素的分析")
    management_suggestion: str = Field(..., description="管理和防治建议")
    # xai_explanation_url: Optional[str] = Field(None, description="指向XAI解释图的URL (未来功能)")