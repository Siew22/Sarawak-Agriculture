# app/services/weather_service.py
import httpx
from typing import Dict, Any

class WeatherService:
    def __init__(self):
        self.api_url = "https://api.open-meteo.com/v1/forecast"

    async def get_current_weather(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """
        根据经纬度从Open-Meteo获取当前天气数据。
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m", # 我们只需要这两项
            "timezone": "auto"
        }
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.api_url, params=params)
                response.raise_for_status()  # 如果请求失败 (例如 4xx or 5xx), 会抛出异常
                data = response.json()
                
                # 提取我们需要的数据
                current_weather = data.get("current", {})
                temperature = current_weather.get("temperature_2m")
                humidity = current_weather.get("relative_humidity_2m")

                if temperature is None or humidity is None:
                    raise ValueError("天气API返回的数据不完整")

                return {"temperature": temperature, "humidity": humidity}

        except httpx.HTTPStatusError as e:
            print(f"天气API请求失败: {e.response.status_code} - {e.response.text}")
            raise ConnectionError("无法连接到天气服务")
        except Exception as e:
            print(f"获取天气数据时发生未知错误: {e}")
            raise ConnectionError("获取天气数据时出错")

# 创建一个全局实例
weather_service = WeatherService()