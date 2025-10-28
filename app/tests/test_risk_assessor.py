# tests/test_risk_assessor.py
from app.models.risk_assessor import risk_assessor

def test_high_risk_scenario():
    """测试在高温高湿下，风险评估是否为高风险"""
    temp = 35.0
    humidity = 95.0
    result = risk_assessor.assess(temp, humidity)
    assert result.risk_level == "High"
    assert result.risk_score > 7.0

def test_low_risk_scenario():
    """测试在低温低湿下，风险评估是否为低风险"""
    temp = 20.0
    humidity = 50.0
    result = risk_assessor.assess(temp, humidity)
    assert result.risk_level == "Low"
    assert result.risk_score < 4.0