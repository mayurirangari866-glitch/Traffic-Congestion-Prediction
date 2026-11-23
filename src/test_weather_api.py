
import requests

url = "https://api.open-meteo.com/v1/forecast"

params = {
    "latitude": 21.1458,
    "longitude": 79.0882,
    "current": (
        "temperature_2m,"
        "relative_humidity_2m,"
        "precipitation,"
        "rain,"
        "weather_code,"
        "wind_speed_10m"
    ),
    "timezone": "Asia/Kolkata"
}

try:
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()

    data = response.json()
    weather = data["current"]

    print("=" * 45)
    print("LIVE NAGPUR WEATHER DATA")
    print("=" * 45)
    print("Time:", weather["time"])
    print("Temperature:", weather["temperature_2m"], "°C")
    print("Humidity:", weather["relative_humidity_2m"], "%")
    print("Precipitation:", weather["precipitation"], "mm")
    print("Rain:", weather["rain"], "mm")
    print("Weather Code:", weather["weather_code"])
    print("Wind Speed:", weather["wind_speed_10m"], "km/h")
    print("=" * 45)
    print("Weather API connection successful!")

except requests.exceptions.RequestException as error:
    print("Weather API connection failed:", error)
except (KeyError, ValueError) as error:
    print("Invalid weather API response:", error)
