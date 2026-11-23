
import requests
import csv
import time
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = BASE_DIR / "outputs" / "live_weather_data.csv"

URL = "https://api.open-meteo.com/v1/forecast"

PARAMS = {
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

INTERVAL_SECONDS = 10  # Collect every 5 minutes

def collect_weather():
    response = requests.get(URL, params=PARAMS, timeout=20)
    response.raise_for_status()

    weather = response.json()["current"]

    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "weather_time": weather["time"],
        "latitude": 21.1458,
        "longitude": 79.0882,
        "temperature_c": weather["temperature_2m"],
        "humidity_percent": weather["relative_humidity_2m"],
        "precipitation_mm": weather["precipitation"],
        "rain_mm": weather["rain"],
        "weather_code": weather["weather_code"],
        "wind_speed_kmph": weather["wind_speed_10m"]
    }

def save_weather(record):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    file_exists = OUTPUT_FILE.exists() and OUTPUT_FILE.stat().st_size > 0

    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=record.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(record)

print("=" * 45)
print("LIVE NAGPUR WEATHER DATA COLLECTION")
print("Collection interval: 10 seconds")
print("Output:", OUTPUT_FILE)
print("Stop using the PyCharm Stop button.")
print("=" * 45)

reading = 0

try:
    while True:
        try:
            record = collect_weather()
            save_weather(record)
            reading += 1

            print("\nReading:", reading)
            print("Time:", record["weather_time"])
            print("Temperature:", record["temperature_c"], "°C")
            print("Humidity:", record["humidity_percent"], "%")
            print("Rain:", record["rain_mm"], "mm")
            print("Saved successfully.")

        except requests.exceptions.RequestException as error:
            print("Weather API error:", error)

        if reading == 3:
            print("\nThree readings collected. Stopping.")
            break

        time.sleep(INTERVAL_SECONDS)

except KeyboardInterrupt:
    print("\nCollection stopped by user.")

print("Total successful readings:", reading)
