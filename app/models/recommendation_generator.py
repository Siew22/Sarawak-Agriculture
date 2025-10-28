from ..schemas.diagnosis import PredictionResult, RiskAssessment, FullDiagnosisReport
from ..services.knowledge_base_service import kb_service

class DynamicRecommendationGenerator:
    def __init__(self):
        """
        在构造函数中初始化一个i18n（国际化）词典，用于存储所有UI标签文本。
        这使得所有面向用户的文本都集中在一个地方，易于管理和翻译。
        """
        self.i18n = {
            "report_title": {"en": "Crop Health Diagnosis Report", "ms": "Laporan Diagnosis Kesihatan Tanaman", "zh": "作物健康诊断报告"},
            "diagnosis_summary_label": {"en": "Diagnosis Summary:", "ms": "Ringkasan Diagnosis:", "zh": "诊断摘要:"},
            "confidence_label": {"en": "Model Confidence", "ms": "Keyakinan Model", "zh": "模型置信度"},
            "env_risk_analysis_label": {"en": "Environmental Risk Analysis:", "ms": "Analisis Risiko Persekitaran:", "zh": "环境风险分析:"},
            "risk_level_label": {"en": "Risk", "ms": "Risiko", "zh": "风险"},
            "risk_score_label": {"en": "Score", "ms": "Skor", "zh": "评分"},
            "unknown_condition_title": {"en": "Unknown Condition Report", "ms": "Laporan Keadaan Tidak Diketahui", "zh": "未知状况诊断报告"},
            "unknown_summary_prefix": {"en": "The model identified a condition with the code", "ms": "Model telah mengenal pasti keadaan dengan kod", "zh": "模型识别出一个代号为"},
            "unknown_summary_suffix": {"en": "but no detailed information is available in our knowledge base. Please consult an expert.", "ms": "tetapi maklumat terperinci tidak tersedia dalam pangkalan pengetahuan kami. Sila rujuk pakar.", "zh": "但知识库中暂无详细信息。请咨询专家。"}
        }

    def _(self, key: str, lang: str) -> str:
        """
        一个简单的内部翻译函数。
        根据给定的key和语言代码(lang)，从i18n词典中查找文本。
        如果找不到指定语言，会回退到英文(en)。
        如果连英文都找不到，会返回key本身，便于调试。
        """
        return self.i18n.get(key, {}).get(lang, self.i18n.get(key, {}).get("en", f"[{key}]"))

    def generate(self, prediction: PredictionResult, risk: RiskAssessment, lang: str = 'en') -> FullDiagnosisReport:
        disease_key = prediction.disease
        disease_info = kb_service.get_disease_info(disease_key)

        if not disease_info:
            # 如果在YAML知识库中找不到对应的病害条目，则生成一份专业的默认报告。
            return self._generate_default_report(prediction, risk, lang)

        # --- 彻底动态化的报告构建逻辑 ---
        name_map = disease_info.get("name", {})
        name = name_map.get(lang, name_map.get("en", disease_key))
        title = f"{self._('report_title', lang)} ({name})"
        
        # 构建诊断摘要，所有标签都通过翻译函数获取
        summary_map = disease_info.get("summary", {})
        summary = summary_map.get(lang, summary_map.get("en", "No detailed description."))
        diagnosis_summary = f"{self._('diagnosis_summary_label', lang)} {summary} ({self._('confidence_label', lang)}: {prediction.confidence:.2%})"
        
        # 构建环境风险分析，所有标签都通过翻译函数获取
        risk_text_map = {
            "High": {
                "en": "The current hot and humid conditions are highly favorable for the development of this disease. Please take immediate preventative action.",
                "ms": "Keadaan panas dan lembap semasa sangat sesuai untuk perkembangan penyakit ini. Sila ambil tindakan pencegahan dengan segera.",
                "zh": "当前炎热潮湿的环境极易诱发此类病害的发生和蔓延，请务必加强防范。"
            },
            "Medium": {
                "en": "The warm and humid environment may encourage disease development. Stay vigilant.",
                "ms": "Persekitaran yang hangat dan lembap mungkin menggalakkan perkembangan penyakit. Sentiasa berwaspada.",
                "zh": "温暖湿润的环境可能诱发病害，请保持警惕。"
            },
            "Low": {
                "en": "Current environmental conditions are relatively stable, but continue to monitor for changes.",
                "ms": "Keadaan persekitaran semasa agak stabil, tetapi terus pantau sebarang perubahan.",
                "zh": "当前环境条件相对有利，但仍需注意天气变化。"
            }
        }
        risk_text = risk_text_map.get(risk.risk_level, {}).get(lang, "")
        environmental_context = (f"{self._('env_risk_analysis_label', lang)} {self._('risk_level_label', lang)} {risk.risk_level} "
                                 f"({self._('risk_score_label', lang)}: {risk.risk_score:.1f}/10). {risk_text}")
        
        # 构建管理建议 (这部分已经是动态的)
        management_suggestion = ""
        treatments = disease_info.get("treatments", [])
        if treatments:
            for treatment in treatments:
                title_map = treatment.get("title", {})
                treatment_title = title_map.get(lang, title_map.get("en", "Suggestion"))
                management_suggestion += f"\n--- {treatment_title} ---\n"
                
                steps = treatment.get("steps", [])
                if steps:
                    for i, step in enumerate(steps):
                        step_map = step if isinstance(step, dict) else {"en": str(step)}
                        step_text = step_map.get(lang, step_map.get("en", ""))
                        management_suggestion += f"{i+1}. {step_text}\n"
        else:
            management_suggestion = self._get_default_suggestion(lang)

        return FullDiagnosisReport(
            title=title,
            diagnosis_summary=diagnosis_summary,
            environmental_context=environmental_context,
            management_suggestion=management_suggestion.strip()
        )

    def _get_default_suggestion(self, lang: str) -> str:
        """获取通用的“咨询专家”建议的多语言版本"""
        suggestions = {
            "en": "You are strongly advised to consult with a local agricultural expert for an accurate diagnosis.",
            "ms": "Anda amat dinasihatkan untuk berunding dengan pakar pertanian tempatan untuk diagnosis yang tepat.",
            "zh": "强烈建议您立即联系本地农业专家进行确认。"
        }
        return suggestions.get(lang, suggestions["en"])

    def _generate_default_report(self, prediction: PredictionResult, risk: RiskAssessment, lang: str) -> FullDiagnosisReport:
        """当在知识库中找不到Key时，生成一份完全动态的、专业的默认报告"""
        title = self._('unknown_condition_title', lang)
        summary_prefix = self._('unknown_summary_prefix', lang)
        summary_suffix = self._('unknown_summary_suffix', lang)
        
        diagnosis_summary = f"{summary_prefix} '{prediction.disease}' ({self._('confidence_label', lang)}: {prediction.confidence:.2%}), {summary_suffix}"
        
        environmental_context = f"{self._('env_risk_analysis_label', lang)} {self._('risk_level_label', lang)} {risk.risk_level} ({self._('risk_score_label', lang)}: {risk.risk_score:.1f}/10)."
        
        return FullDiagnosisReport(
            title=title,
            diagnosis_summary=diagnosis_summary,
            environmental_context=environmental_context,
            management_suggestion=self._get_default_suggestion(lang)
        )

# 创建一个全局实例，供main.py导入和使用
report_generator_v2 = DynamicRecommendationGenerator()