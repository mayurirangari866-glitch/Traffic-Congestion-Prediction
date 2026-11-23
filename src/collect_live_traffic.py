
import os
import csv
import time
import requests

from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

# Project configuration
PROJECT_FOLDER = Path(__file__).resolve().parent.parent
OUTPUT_FOLDER = PROJECT_FOLDER / "outputs"
OUTPUT_FOLDER.mkdir(exist_ok=True)

load_dotenv(PROJECT_FOLDER / ".env")
API_KEY = os.getenv("TOMTOM_API_KEY")

if not API_KEY:
    raise ValueError("TomTom API key not found.")

# Nagpur test coordinates
LATITUDE = 21.1458
LONGITUDE = 79.0882

API_URL = (
    "https://api.tomtom.com/traffic/services/4/"
    "flowSegmentData/relative/10/json"
)

CSV_FILE = OUTPUT_FOLDER / "live_traffic_data.csv"

INTERVAL_SECONDS = 60

def collect_traffic():

    params = {
        "point": f"{LATITUDE},{LONGITUDE}",
        "unit": "KMPH",
        "key": API_KEY
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()
    flow = data["flowSegmentData"]

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "current_speed_kmph": flow.get("currentSpeed"),
        "free_flow_speed_kmph": flow.get("freeFlowSpeed"),
        "confidence": flow.get("confidence"),
        "road_closure": flow.get("roadClosure")
    }


def save_traffic(record):

    file_exists = CSV_FILE.exists() and CSV_FILE.stat().st_size > 0

    with open(
        CSV_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=record.keys()
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(record)


def main():

    print("=" * 55)
    print("CONTINUOUS NAGPUR TRAFFIC DATA COLLECTION")
    print("=" * 55)

    print("Collection interval:", INTERVAL_SECONDS, "seconds")
    print("Output file:", CSV_FILE)
    print("Press CTRL+C to stop collection.")

    count = 0

    try:

        while True:

            try:
                record = collect_traffic()
                save_traffic(record)

                count += 1

                print("\nReading:", count)
                print("Time:", record["timestamp_utc"])
                print("Current speed:",
                      record["current_speed_kmph"], "km/h")
                print("Free-flow speed:",
                      record["free_flow_speed_kmph"], "km/h")
                print("Confidence:", record["confidence"])
                print("Saved successfully.")

            except requests.exceptions.RequestException as error:
                print("\nAPI connection error:", error)

            except (KeyError, ValueError) as error:
                print("\nUnexpected API response:", error)

            time.sleep(INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n\nCollection stopped by user.")
        print("Total successful readings:", count)
        print("CSV saved at:", CSV_FILE)


if __name__ == "__main__":
    main()
