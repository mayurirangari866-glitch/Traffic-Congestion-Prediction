
import os
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load API key
PROJECT_FOLDER = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_FOLDER / ".env")

API_KEY = os.getenv("TOMTOM_API_KEY")

if not API_KEY:
    print("ERROR: API key not found.")
    raise SystemExit

print("API key loaded successfully.")
print("\nConnecting to TomTom Traffic API...")

url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/relative/10/json"

params = {
    "point": "21.1458,79.0882",
    "unit": "KMPH",
    "key": API_KEY
}

try:
    response = requests.get(url, params=params, timeout=30)

    print("HTTP Status:", response.status_code)

    if response.status_code == 200:
        data = response.json()
        flow = data.get("flowSegmentData", {})

        print("\nSUCCESS: TOMTOM TRAFFIC DATA RECEIVED")
        print("Location: Nagpur test coordinates")
        print("Current Speed:", flow.get("currentSpeed"), "km/h")
        print("Free Flow Speed:", flow.get("freeFlowSpeed"), "km/h")
        print("Confidence:", flow.get("confidence"))
        print("Road Closure:", flow.get("roadClosure"))

    else:
        print("\nAPI REQUEST FAILED")
        print("Error details:", response.text[:500])

except requests.exceptions.RequestException as error:
    print("Connection error:", error)

