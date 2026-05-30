import serial
import csv
from datetime import datetime

ser = serial.Serial('COM6', 115200)

filename = "sensor_data.csv"

with open(filename, "a", newline="") as file:
    writer = csv.writer(file)

    while True:
        try:
            line = ser.readline().decode(errors="ignore").strip()

            parts = line.split(",")

            if len(parts) == 5:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                millis = parts[0]
                temp = parts[1]
                hum = parts[2]
                soil = parts[3]
                ph = parts[4]

                print(f"pH value: {ph}")
                writer.writerow([timestamp, temp, hum, soil, ph])
                file.flush()

        except Exception as e:
            print("Error:", e)