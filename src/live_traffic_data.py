
import os
import json
import requests
import csv

from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

# Project paths
PROJECT_FOLDER = Path(__file__).resolve().parent.parent
OUTPUT_FOLDER = PROJECT_FOLDER / "outputs"
OUTPUT_FOLDER.mkdir(exist_ok=True)

# Load API key
load_dotenv(PROJECT_FOLDER / ".env")
API_KEY = os.getenv("TOMTOM_API_KEY")

if not API_KEY:
    raise ValueError("TomTom API key not found.")

# Nagpur test coordinates
LATITUDE = 21.1458
LONGITUDE = 79.0882

URL = (
    "https://api.tomtom.com/traffic/services/4/"
    "flowSegmentData/relative/10/json"
)

params = {
    "point": f"{LATITUDE},{LONGITUDE}",
    "unit": "KMPH",
    "key": API_KEY
}

print("FETCHING LIVE NAGPUR TRAFFIC DATA")
print("=" * 45)

try:
    response = requests.get(URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    flow = data["flowSegmentData"]

    timestamp = datetime.now(timezone.utc).isoformat()

    record = {
        "timestamp_utc": timestamp,
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "current_speed_kmph": flow.get("currentSpeed"),
        "free_flow_speed_kmph": flow.get("freeFlowSpeed"),
        "confidence": flow.get("confidence"),
        "road_closure": flow.get("roadClosure")
    }

    # Save raw API response
    with open(
        OUTPUT_FOLDER / "live_traffic_data.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(data, file, indent=4)

    # Append collected record to CSV
    csv_path = OUTPUT_FOLDER / "live_traffic_data.csv"

    with open(
        csv_path,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=record.keys()
        )

        if file.tell() == 0:
            writer.writeheader()

        writer.writerow(record)

    print("\nLIVE TRAFFIC DATA RECEIVED")
    print("-" * 45)

    for key, value in record.items():
        print(f"{key}: {value}")

    print("\nData saved successfully.")
    print("CSV:", csv_path)
    print("JSON:", OUTPUT_FOLDER / "live_traffic_data.json")

except requests.exceptions.RequestException as error:
    print("API connection error:", error)

except (KeyError, ValueError) as error:
    print("Unexpected API response:", error)

print("\nCOLLECTION TEST FINISHED")
