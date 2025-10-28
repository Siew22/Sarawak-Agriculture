import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from ..schemas.diagnosis import RiskAssessment

class FuzzyRiskAssessor:
    def __init__(self):
        # 定义输入变量
        temperature = ctrl.Antecedent(np.arange(15, 41, 1), 'temperature')
        humidity = ctrl.Antecedent(np.arange(40, 101, 1), 'humidity')
        
        # 定义输出变量
        disease_risk = ctrl.Consequent(np.arange(0, 11, 1), 'disease_risk')

        # 定义隶属函数 (语言化变量)
        temperature['cool'] = fuzz.trapmf(temperature.universe, [15, 15, 22, 26])
        temperature['warm'] = fuzz.trimf(temperature.universe, [24, 28, 32])
        temperature['hot'] = fuzz.trapmf(temperature.universe, [30, 34, 40, 40])

        humidity['low'] = fuzz.trapmf(humidity.universe, [40, 40, 55, 65])
        humidity['medium'] = fuzz.trimf(humidity.universe, [60, 75, 90])
        humidity['high'] = fuzz.trapmf(humidity.universe, [85, 95, 100, 100])

        disease_risk['low'] = fuzz.trimf(disease_risk.universe, [0, 2, 4])
        disease_risk['medium'] = fuzz.trimf(disease_risk.universe, [3, 5, 7])
        disease_risk['high'] = fuzz.trimf(disease_risk.universe, [6, 8, 10])

        # 定义专家规则库
        self.rules = [
            ctrl.Rule(temperature['hot'] & humidity['high'], disease_risk['high']),
            ctrl.Rule(temperature['hot'] & humidity['medium'], disease_risk['medium']),
            ctrl.Rule(temperature['warm'] & humidity['high'], disease_risk['high']),
            ctrl.Rule(temperature['warm'] & humidity['medium'], disease_risk['medium']),
            ctrl.Rule(humidity['low'], disease_risk['low']),
            ctrl.Rule(temperature['cool'], disease_risk['low'])
        ]
        
        risk_ctrl_system = ctrl.ControlSystem(self.rules)
        self.risk_simulator = ctrl.ControlSystemSimulation(risk_ctrl_system)

    def assess(self, temp_value: float, humidity_value: float) -> RiskAssessment:
        self.risk_simulator.input['temperature'] = temp_value
        self.risk_simulator.input['humidity'] = humidity_value
        self.risk_simulator.compute()
        
        score = self.risk_simulator.output['disease_risk']
        level = "Low"
        if score > 7.0:
            level = "High"
        elif score > 4.0:
            level = "Medium"
            
        return RiskAssessment(risk_score=score, risk_level=level)

# 创建一个全局实例
risk_assessor = FuzzyRiskAssessor()