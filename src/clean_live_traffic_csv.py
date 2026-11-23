import csv
import shutil
from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent

csv_path = project_folder / "outputs" / "live_traffic_multi_location.csv"
backup_path = project_folder / "outputs" / "live_traffic_multi_location_backup.csv"

valid_locations = {
    "Existing Nagpur Monitoring Point",
    "Chhatrapati Square - Wardha Road"
}

if not csv_path.exists():
    print("ERROR: Traffic CSV file not found.")
    raise SystemExit

if not backup_path.exists():
    shutil.copy2(csv_path, backup_path)

with open(csv_path, "r", newline="", encoding="utf-8") as file:
    reader = csv.reader(file)
    header = next(reader)
    rows = list(reader)

cleaned_rows = [
    row for row in rows
    if len(row) == len(header)
    and row[1] in valid_locations
]

with open(csv_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(cleaned_rows)

print("CLEANUP SUCCESSFUL")
print("Original records:", len(rows))
print("Retained records:", len(cleaned_rows))
print("Removed records:", len(rows) - len(cleaned_rows))

for location in valid_locations:
    count = sum(row[1] == location for row in cleaned_rows)
    print(location, ":", count)