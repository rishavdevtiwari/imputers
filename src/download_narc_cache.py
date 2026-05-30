import os
import json
import time
import requests
import math

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "local_soil_db.json")

DISTRICTS = {
    "Taplejung": (27.35, 87.66), "Sankhuwasabha": (27.36, 87.21), "Solukhumbu": (27.79, 86.72),
    "Okhaldhunga": (27.31, 86.49), "Khotang": (27.19, 86.77), "Bhojpur": (27.16, 87.04),
    "Dhankuta": (26.98, 87.33), "Terhathum": (27.12, 87.54), "Panchthar": (27.16, 87.75),
    "Ilam": (26.91, 87.92), "Jhapa": (26.638, 87.995), "Morang": (26.643, 87.441),
    "Sunsari": (26.611, 87.149), "Udayapur": (26.90, 86.58), "Saptari": (26.56, 86.74),
    "Siraha": (26.65, 86.20), "Dhanusha": (26.790, 85.932), "Mahottari": (26.85, 85.79),
    "Sarlahi": (26.98, 85.55), "Sindhuli": (27.24, 85.95), "Ramechhap": (27.36, 86.08),
    "Dolakha": (27.81, 86.15), "Sindhupalchok": (27.95, 85.73), "Kavrepalanchok": (27.531, 85.553),
    "Lalitpur": (27.65, 85.31), "Bhaktapur": (27.67, 85.42), "Kathmandu": (27.717, 85.324),
    "Nuwakot": (27.92, 85.16), "Rasuwa": (28.11, 85.30), "Dhading": (27.97, 84.90),
    "Makwanpur": (27.42, 85.03), "Rautahat": (26.98, 85.32), "Bara": (27.022, 85.011),
    "Parsa": (27.042, 84.872), "Chitwan": (27.533, 84.416), "Gorkha": (28.27, 84.72),
    "Lamjung": (28.23, 84.41), "Tanahun": (27.96, 84.23), "Syangja": (28.08, 83.84),
    "Kaski": (28.252, 83.972), "Manang": (28.55, 84.02), "Mustang": (28.801, 83.824),
    "Myagdi": (28.34, 83.56), "Parbat": (28.21, 83.68), "Baglung": (28.27, 83.58),
    "Gulmi": (28.06, 83.25), "Palpa": (27.86, 83.55), "Nawalparasi East": (27.653, 84.124),
    "Nawalparasi West": (27.53, 83.74), "Rupandehi": (27.575, 83.454), "Kapilvastu": (27.545, 83.053),
    "Arghakhanchi": (27.93, 83.09), "Pyuthan": (28.11, 82.86), "Rolpa": (28.32, 82.63),
    "Rukum East": (28.57, 82.80), "Rukum West": (28.65, 82.47), "Salyan": (28.38, 82.15),
    "Dang": (28.02, 82.32), "Banke": (28.121, 81.674), "Bardiya": (28.31, 81.33),
    "Surkhet": (28.591, 81.633), "Dailekh": (28.84, 81.71), "Jajarkot": (28.86, 82.19),
    "Dolpa": (29.13, 83.05), "Jumla": (29.275, 82.184), "Kalikot": (29.16, 81.60),
    "Mugu": (29.56, 82.12), "Humla": (29.96, 81.82), "Bajura": (29.49, 81.56),
    "Bajhang": (29.62, 81.20), "Achham": (29.11, 81.29), "Doti": (29.26, 80.95),
    "Kailali": (28.715, 80.584), "Kanchanpur": (28.903, 80.222), "Dadeldhura": (29.29, 80.59),
    "Baitadi": (29.51, 80.52), "Darchula": (29.83, 80.60)
}

def fetch_narc_data(lat, lon):
    url = f"https://soil.narc.gov.np/soil/api/soildata?lat={lat}&lon={lon}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        response = requests.get(url, headers=headers, timeout=5.0)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            return data[0]
        elif isinstance(data, dict):
            for key in ("results", "data", "soildata", "soil"):
                nested = data.get(key)
                if isinstance(nested, list) and nested:
                    return nested[0]
                if isinstance(nested, dict):
                    return nested
            return data
    except Exception as e:
        return None
    return None

def generate_spiral_offsets(max_radius=0.12, step=0.015, limit=30):
    """Generate up to 'limit' coordinate offsets in a spiral grid, sorted by distance."""
    offsets = []
    grid_limit = int(max_radius / step) + 1
    for dx in range(-grid_limit, grid_limit + 1):
        for dy in range(-grid_limit, grid_limit + 1):
            x = dx * step
            y = dy * step
            dist = math.hypot(x, y)
            if dist <= max_radius:
                offsets.append((x, y, dist))
    # Sort by distance to prioritize points closer to center
    offsets.sort(key=lambda item: item[2])
    # Extract only the x, y tuples and limit to 30
    return [(item[0], item[1]) for item in offsets][:limit]

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    db = {}
    
    # Pre-calculate our top 30 intelligent spiral shifts
    shifts = generate_spiral_offsets(max_radius=0.12, step=0.015, limit=30)
    
    print(f"Generating authentic NARC local cache for {len(DISTRICTS)} districts...")
    
    for district, (base_lat, base_lon) in DISTRICTS.items():
        found = False
        print(f"[{district}] Testing base coordinates ({base_lat:.3f}, {base_lon:.3f})...", end=" ")
        
        for idx, (dlat, dlon) in enumerate(shifts):
            lat = base_lat + dlat
            lon = base_lon + dlon
            payload = fetch_narc_data(lat, lon)
            
            # Check if payload exists and has genuine soil values
            if payload and payload.get("result") != "Please select the crop land" and "ph" in payload:
                payload["district"] = district
                payload["lat"] = lat
                payload["lon"] = lon
                payload["source"] = "live"
                db[district] = payload
                found = True
                if idx == 0:
                    print("OK!")
                else:
                    print(f"Shift {idx}/30 ({dlat:+.3f}, {dlon:+.3f}) -> Authentic Data Captured!")
                break
            elif payload and payload.get("result") == "Please select the crop land":
                pass # Keep spiraling
                
            time.sleep(0.1) # Be nice to the API
        
        if not found:
            print(f"Warning: Exhausted 30 offset attempts for {district}. No authentic data found.")
            # Depending on constraints, we can still fall back if 30 searches fail
            db[district] = {"district": district, "lat": base_lat, "lon": base_lon, "source": "exhausted"}
            
    with open(OUTPUT_PATH, "w") as f:
        json.dump(db, f, indent=4)
        
    print(f"\\nSuccessfully saved {len(db)} district profiles to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
