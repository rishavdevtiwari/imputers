import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from rapidfuzz import fuzz, process
import joblib
import json
from datetime import datetime, timedelta
import os
import time
import re

# ============================================================================
# PART 1: LOAD YOUR YIELD PREDICTION MODEL
# ============================================================================

def load_yield_model():
    """Load the pre-trained yield prediction model and related files"""
    try:
        model = joblib.load('models/best_yield_prediction_model.joblib')
        scaler = joblib.load('models/feature_scaler.joblib')
        label_encoders = joblib.load('models/label_encoders.joblib')
        feature_columns = joblib.load('models/feature_columns.joblib')
        print("[OK] Yield prediction model loaded successfully")
        return model, scaler, label_encoders, feature_columns
    except FileNotFoundError:
        print("[WARNING] Model files not found. Please run yield.py first to train the model.")
        print("         Running in price-only mode (profit calculation will use estimated yields)")
        return None, None, None, None

# ============================================================================
# PART 2: PRICE STORAGE WITH FALLBACK MECHANISM
# ============================================================================

class PriceStorage:
    """Handles saving, loading, and fallback for price data"""
    
    def __init__(self, filename='crop_prices.txt'):
        self.filename = filename
        self.price_history = []
        self.load_price_history()
    
    def load_price_history(self):
        """Load all price records from the text file"""
        if not os.path.exists(self.filename):
            print(f"[INFO] No price file found at {self.filename}. Will create new one.")
            return
        
        try:
            with open(self.filename, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse records
            records = content.split("="*80)
            for record in records:
                if record.strip():
                    parsed_record = self._parse_price_record(record)
                    if parsed_record:
                        self.price_history.append(parsed_record)
            
            print(f"[OK] Loaded {len(self.price_history)} historical price records")
        except Exception as e:
            print(f"[WARNING] Could not load price history: {e}")
    
    def _parse_price_record(self, record):
        """Parse a price record from text format to dictionary"""
        try:
            lines = record.strip().split('\n')
            if not lines:
                return None
            
            # Extract timestamp
            timestamp_line = lines[0] if lines else ""
            timestamp_match = re.search(r'PRICE RECORD - (.+)', timestamp_line)
            timestamp = timestamp_match.group(1) if timestamp_match else None
            
            # Check if it's a specific crop record or full price list
            if "CROP:" in record:
                # Parse specific crop record
                crop_match = re.search(r'CROP: (.+)', record)
                crop_name = crop_match.group(1).strip() if crop_match else None
                
                price_match = re.search(r'Price per kg: NPR ([\d.]+)', record)
                price = float(price_match.group(1)) if price_match else None
                
                profit_match = re.search(r'NET PROFIT: NPR ([\d,]+\.?\d*)', record)
                profit = float(profit_match.group(1).replace(',', '')) if profit_match else None
                
                return {
                    'type': 'crop_specific',
                    'timestamp': timestamp,
                    'crop': crop_name,
                    'price_per_kg': price,
                    'net_profit': profit,
                    'datetime': datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S") if timestamp else None
                }
            else:
                # Parse full price list
                prices = {}
                price_lines = [line for line in lines if 'kg' in line or 'quintal' in line]
                
                for line in price_lines:
                    parts = line.split()
                    if len(parts) >= 5:
                        crop = parts[0].strip()
                        try:
                            min_price = float(parts[1])
                            max_price = float(parts[2])
                            avg_price = float(parts[3])
                            unit = parts[4]
                            
                            prices[crop] = {
                                'min_price': min_price,
                                'max_price': max_price,
                                'avg_price': avg_price,
                                'unit': unit
                            }
                        except ValueError:
                            continue
                
                return {
                    'type': 'full_list',
                    'timestamp': timestamp,
                    'prices': prices,
                    'datetime': datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S") if timestamp else None
                }
        except Exception as e:
            return None
    
    def save_prices(self, prices, crop_name=None, profit_data=None):
        """
        Save current prices to a text file with timestamp.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.filename, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"PRICE RECORD - {timestamp}\n")
            f.write("="*80 + "\n")
            
            if crop_name and profit_data:
                # Save specific crop with profit info
                f.write(f"\nCROP: {crop_name.upper()}\n")
                f.write("-"*40 + "\n")
                f.write(f"Price per kg: NPR {profit_data.get('price_per_kg', 'N/A')}\n")
                f.write(f"Price Source: {profit_data.get('price_source', 'Unknown')}\n")
                f.write(f"Price Age: {profit_data.get('price_age', 'N/A')}\n")
                f.write(f"Predicted Yield: {profit_data.get('yield_kg_per_ha', 'N/A'):.0f} kg/ha\n")
                f.write(f"Total Revenue: NPR {profit_data.get('total_revenue_npr', 'N/A'):.2f}\n")
                f.write(f"Production Cost: NPR {profit_data.get('production_cost_npr', 'N/A'):.2f}\n")
                f.write(f"NET PROFIT: NPR {profit_data.get('net_profit_npr', 'N/A'):.2f}\n")
                f.write(f"ROI: {profit_data.get('roi_percent', 'N/A'):.1f}%\n")
            else:
                # Save all prices
                f.write("\nCROP PRICES:\n")
                f.write("-"*40 + "\n")
                f.write(f"{'Crop':<20} {'Min':<10} {'Max':<10} {'Avg':<10} {'Unit'}\n")
                f.write("-"*60 + "\n")
                
                for crop, data in prices.items():
                    f.write(f"{crop:<20} {data['min_price']:<10} {data['max_price']:<10} {data['avg_price']:<10} {data['unit']}\n")
            
            f.write("\n" + "="*80 + "\n")
        
        # Update history
        self.load_price_history()  # Reload to update history
        print(f"[OK] Prices saved to {self.filename}")
    
    def get_latest_prices(self, max_age_hours=168):  # Default 7 days
        """
        Get the most recent full price list from storage.
        
        Args:
            max_age_hours: Maximum age in hours (default 168 = 7 days)
        
        Returns:
            Dictionary of prices or None if no recent data
        """
        recent_records = [r for r in self.price_history if r['type'] == 'full_list']
        
        if not recent_records:
            print("[INFO] No historical price data found")
            return None
        
        # Sort by datetime (newest first)
        recent_records.sort(key=lambda x: x['datetime'] if x['datetime'] else datetime.min, reverse=True)
        latest = recent_records[0]
        
        if latest['datetime']:
            age = datetime.now() - latest['datetime']
            if age.total_seconds() / 3600 <= max_age_hours:
                print(f"[OK] Using stored prices from {latest['timestamp']} (age: {age.days} days, {age.seconds//3600} hours)")
                return latest['prices']
            else:
                print(f"[WARNING] Stored prices are too old: {age.days} days old (max: {max_age_hours//24} days)")
                return None
        
        return latest.get('prices', None)
    
    def get_latest_price_for_crop(self, crop_name, max_age_hours=168):
        """
        Get the most recent price for a specific crop.
        
        Args:
            crop_name: Name of the crop
            max_age_hours: Maximum age in hours
        
        Returns:
            Price data dictionary or None
        """
        crop_lower = crop_name.lower()
        
        # First try to get from full price lists
        full_price_lists = [r for r in self.price_history if r['type'] == 'full_list']
        full_price_lists.sort(key=lambda x: x['datetime'] if x['datetime'] else datetime.min, reverse=True)
        
        for price_list in full_price_lists:
            if price_list['datetime']:
                age = datetime.now() - price_list['datetime']
                if age.total_seconds() / 3600 <= max_age_hours:
                    prices = price_list['prices']
                    if crop_lower in prices:
                        print(f"[OK] Found price for '{crop_name}' in stored data from {price_list['timestamp']}")
                        return prices[crop_lower]
        
        # If not found, try crop-specific records
        crop_records = [r for r in self.price_history if r['type'] == 'crop_specific' and r.get('crop', '').lower() == crop_lower]
        crop_records.sort(key=lambda x: x['datetime'] if x['datetime'] else datetime.min, reverse=True)
        
        if crop_records:
            latest = crop_records[0]
            if latest['datetime']:
                age = datetime.now() - latest['datetime']
                if age.total_seconds() / 3600 <= max_age_hours:
                    print(f"[OK] Found price for '{crop_name}' in crop-specific record from {latest['timestamp']}")
                    return {
                        'min_price': latest['price_per_kg'],
                        'max_price': latest['price_per_kg'],
                        'avg_price': latest['price_per_kg'],
                        'unit': 'kg'
                    }
        
        print(f"[WARNING] No recent stored price found for '{crop_name}'")
        return None
    
    def load_latest_prices(self):
        """Load and display the latest price record"""
        if not self.price_history:
            print(f"[INFO] No price records found")
            return None
        
        latest = self.price_history[0]
        print(f"\n[LATEST PRICE RECORD - {latest['timestamp']}]")
        
        if latest['type'] == 'crop_specific':
            print(f"Crop: {latest.get('crop', 'Unknown')}")
            print(f"Price: NPR {latest.get('price_per_kg', 'N/A')}/kg")
            if latest.get('net_profit'):
                print(f"Profit: NPR {latest['net_profit']:,.2f}")
        else:
            print(f"\n{'Crop':<20} {'Min':<10} {'Max':<10} {'Avg':<10} {'Unit'}")
            print("-"*60)
            for crop, data in latest.get('prices', {}).items():
                print(f"{crop:<20} {data['min_price']:<10} {data['max_price']:<10} {data['avg_price']:<10} {data['unit']}")
        
        return latest


# ============================================================================
# PART 3: KALIMATI PRICE SCRAPER WITH FALLBACK
# ============================================================================

class KalimatiPriceScraper:
    """Scraper for fetching vegetable and crop prices from Kalimati market with fallback to stored prices"""
    
    BASE_URL = "https://www.kalimatimarket.gov.np/"  # Replace with actual Kalimati market URL
    
    CROP_NAME_MAPPINGS = {
        'paddy': 'rice',
        'maize': 'corn',
        'millet': 'kodo',
        'buckwheat': 'phapar',
        'potato': 'alu',
        'tomato': 'tamatar',
        'cauliflower': 'kauli',
        'cabbage': 'bandakopi',
        'onion': 'pyaj',
        'garlic': 'lasun',
        'ginger': 'aduwa',
        'spinach': 'palungo',
        'radish': 'mula',
        'carrot': 'gajar',
        'beans': 'simi',
        'lentil': 'masuro',
        'rice': 'dhan',
        'wheat': 'gahun'
    }
    
    def __init__(self, price_storage):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.price_cache = {}
        self.price_storage = price_storage
        self.last_fetch_time = None
        self.last_fetch_success = False
        
    def fetch_prices(self, force_refresh=False):
        """
        Fetch current prices from Kalimati market.
        Falls back to stored prices if live fetch fails.
        
        Args:
            force_refresh: If True, ignores cache and fetches fresh
        """
        # Return cached prices if fresh enough (less than 1 hour old)
        if not force_refresh and self.price_cache and self.last_fetch_time:
            age = datetime.now() - self.last_fetch_time
            if age.total_seconds() < 3600:  # 1 hour cache
                print(f"[OK] Using cached prices (age: {age.seconds//60} minutes)")
                return self.price_cache
        
        print("\n[INFO] Attempting to fetch live prices from Kalimati market...")
        
        try:
            import urllib3
            urllib3.disable_warnings()
            response = self.session.get('https://kalimatimarket.gov.np/price', timeout=15, verify=False)
            response.encoding = 'utf-8'
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            tables = soup.find_all('table')
            
            live_prices = {}
            if tables:
                rows = tables[0].find_all('tr')
                nepali_to_eng = {
                    'गोलभेडा': 'tomato', 'आलु': 'potato', 'प्याज': 'onion', 'लसुन': 'garlic',
                    'अदुवा': 'ginger', 'काउली': 'cauliflower', 'बन्दा': 'cabbage',
                    'पालुङ्गो': 'spinach', 'सिमी': 'beans', 'दाल': 'lentil',
                    'धान': 'paddy', 'चामल': 'rice', 'मकै': 'maize', 'कोदो': 'millet',
                    'फापर': 'buckwheat', 'गहुँ': 'wheat', 'जौ': 'barley'
                }
                
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 4:
                        nepali_name = cols[0].text.strip()
                        try:
                            def parse_price(s):
                                s = s.replace('रू.', '').replace('रू', '').replace('Rs.', '').replace('Rs', '').replace(',', '').strip()
                                return float(s)
                            
                            min_p = parse_price(cols[1].text)
                            max_p = parse_price(cols[2].text)
                            avg_p = parse_price(cols[3].text)
                            
                            matched_eng = None
                            for nep, eng in nepali_to_eng.items():
                                if nep in nepali_name:
                                    matched_eng = eng
                                    break
                            
                            if matched_eng:
                                if matched_eng not in live_prices:
                                    live_prices[matched_eng] = {'min_price': min_p, 'max_price': max_p, 'avg_price': avg_p, 'unit': 'kg'}
                                else:
                                    live_prices[matched_eng]['avg_price'] = (live_prices[matched_eng]['avg_price'] + avg_p) / 2
                        except Exception:
                            continue
            
            if live_prices:
                self.price_cache = live_prices
                self.last_fetch_time = datetime.now()
                self.last_fetch_success = True
                
                # Save to storage
                self.price_storage.save_prices(live_prices)
                
                print(f"[SUCCESS] Fetched {len(live_prices)} live crop prices from Kalimati")
                return live_prices
            else:
                raise Exception("No prices returned from scraper")
                
        except Exception as e:
            print(f"[ERROR] Failed to fetch live prices: {e}")
            self.last_fetch_success = False
            
            # FALLBACK: Use stored prices
            print("\n[FALLBACK] Attempting to use stored prices from text file...")
            stored_prices = self.price_storage.get_latest_prices()
            
            if stored_prices:
                print(f"[OK] Using stored prices (last successful fetch: {self.price_storage.price_history[0]['timestamp'] if self.price_storage.price_history else 'Unknown'})")
                self.price_cache = stored_prices
                return stored_prices
            else:
                print("[ERROR] No stored prices available. Using default mock prices as last resort.")
                default_prices = self._get_mock_prices()
                return default_prices
    
    def _get_mock_prices(self):
        """
        Mock price data - replace with actual scraping.
        """
        return {
            'paddy': {'min_price': 3200, 'max_price': 3800, 'avg_price': 3500, 'unit': 'quintal'},
            'rice': {'min_price': 4500, 'max_price': 5500, 'avg_price': 5000, 'unit': 'quintal'},
            'maize': {'min_price': 2800, 'max_price': 3400, 'avg_price': 3100, 'unit': 'quintal'},
            'millet': {'min_price': 2500, 'max_price': 3000, 'avg_price': 2750, 'unit': 'quintal'},
            'buckwheat': {'min_price': 3000, 'max_price': 3600, 'avg_price': 3300, 'unit': 'quintal'},
            'potato': {'min_price': 40, 'max_price': 60, 'avg_price': 50, 'unit': 'kg'},
            'tomato': {'min_price': 50, 'max_price': 80, 'avg_price': 65, 'unit': 'kg'},
            'onion': {'min_price': 45, 'max_price': 70, 'avg_price': 55, 'unit': 'kg'},
            'garlic': {'min_price': 120, 'max_price': 180, 'avg_price': 150, 'unit': 'kg'},
            'ginger': {'min_price': 60, 'max_price': 90, 'avg_price': 75, 'unit': 'kg'},
            'cauliflower': {'min_price': 35, 'max_price': 55, 'avg_price': 45, 'unit': 'kg'},
            'cabbage': {'min_price': 30, 'max_price': 50, 'avg_price': 40, 'unit': 'kg'},
            'spinach': {'min_price': 25, 'max_price': 40, 'avg_price': 32, 'unit': 'kg'},
            'beans': {'min_price': 70, 'max_price': 110, 'avg_price': 90, 'unit': 'kg'},
            'lentil': {'min_price': 120, 'max_price': 150, 'avg_price': 135, 'unit': 'kg'},
            'wheat': {'min_price': 3000, 'max_price': 3600, 'avg_price': 3300, 'unit': 'quintal'},
            'barley': {'min_price': 2600, 'max_price': 3200, 'avg_price': 2900, 'unit': 'quintal'}
        }
    
    def get_price(self, crop_name, use_fuzzy_match=True, fallback_to_stored=True):
        """
        Get price for a crop with multiple fallback levels.
        
        Priority:
        1. Live/cached prices from current session
        2. Stored prices from text file
        3. Default mock prices
        
        Args:
            crop_name: Name of the crop
            use_fuzzy_match: Enable fuzzy matching
            fallback_to_stored: Use stored prices if live fails
        """
        crop_lower = crop_name.lower().strip()
        
        # Level 1: Try current cache (could be live or stored)
        if crop_lower in self.price_cache:
            return self.price_cache[crop_lower]
        
        # Check mappings
        mapped_name = self.CROP_NAME_MAPPINGS.get(crop_lower)
        if mapped_name and mapped_name in self.price_cache:
            return self.price_cache[mapped_name]
        
        # Reverse mapping
        for key, value in self.CROP_NAME_MAPPINGS.items():
            if crop_lower == value and key in self.price_cache:
                return self.price_cache[key]
        
        # Level 2: Fuzzy matching on current cache
        if use_fuzzy_match:
            best_match, score, matched_name = self._fuzzy_match_crop(crop_lower)
            if score >= 80:
                print(f"[INFO] Fuzzy matched '{crop_name}' -> '{matched_name}' (score: {score}%)")
                return self.price_cache[matched_name]
        
        # Level 3: Try stored prices for this specific crop
        if fallback_to_stored:
            stored_price = self.price_storage.get_latest_price_for_crop(crop_lower)
            if stored_price:
                print(f"[OK] Using stored price for '{crop_name}'")
                return stored_price
                
        # Level 4: Fallback to mock prices if all else fails
        mock_prices = self._get_mock_prices()
        if crop_lower in mock_prices:
            print(f"[FALLBACK] Kalimati and storage failed. Using default mock price for '{crop_name}'")
            return mock_prices[crop_lower]
        
        return None
    
    def _fuzzy_match_crop(self, crop_name):
        """Find best matching crop name using fuzzy matching"""
        available_crops = list(self.price_cache.keys())
        
        if not available_crops:
            return None, 0, None
        
        result = process.extractOne(
            crop_name, 
            available_crops,
            scorer=fuzz.WRatio,
            score_cutoff=60
        )
        
        if result:
            matched_name, score, _ = result
            return matched_name, score, matched_name
        
        return None, 0, None
    
    def get_price_per_kg(self, crop_name):
        """Get price per kilogram (converted from quintal if needed) with fallback"""
        price_data = self.get_price(crop_name)
        
        if not price_data:
            return None
        
        avg_price = price_data['avg_price']
        unit = price_data['unit']
        
        # Convert to per kg if needed
        if unit == 'quintal':
            return avg_price / 100  # 1 quintal = 100 kg
        elif unit == 'kg':
            return avg_price
        else:
            return avg_price
    
    def refresh_prices(self):
        """Manually refresh prices from live source"""
        return self.fetch_prices(force_refresh=True)


# ============================================================================
# PART 4: PROFIT CALCULATOR WITH PRICE SOURCE TRACKING
# ============================================================================

class ProfitCalculator:
    """Calculates profit based on predicted yield and market prices"""
    
    def __init__(self, model=None, scaler=None, label_encoders=None, feature_columns=None, price_storage=None):
        self.model = model
        self.scaler = scaler
        self.label_encoders = label_encoders
        self.feature_columns = feature_columns
        self.price_storage = price_storage
        self.price_scraper = KalimatiPriceScraper(price_storage) if price_storage else None
        
        # Production cost estimates (NPR per kg) - Transportation removed
        self.production_costs = {
            'default': 5,
            'paddy': 4,
            'rice': 5,
            'maize': 4,
            'millet': 5,
            'buckwheat': 5,
            'potato': 6,
            'tomato': 8,
            'onion': 7,
            'garlic': 10
        }
        
        # Labor and input costs
        self.labor_cost_per_ha = 25000
        self.input_cost_per_ha = 15000
    
    def predict_yield(self, district, crop_type, area, avg_temp, max_temp, min_temp,
                      humidity, rainfall, solar_radiation, wind_speed, ph_value, fertilizer):
        """Predict yield using the trained model if available"""
        if self.model and self.label_encoders:
            try:
                district_encoded = self.label_encoders['Districts'].transform([district])[0]
                crop_encoded = self.label_encoders['crop_type'].transform([crop_type])[0]
                
                features = np.array([[
                    crop_encoded, district_encoded, ph_value, fertilizer,
                    avg_temp, max_temp, min_temp, humidity, rainfall,
                    solar_radiation, wind_speed, area
                ]])
                
                features_scaled = self.scaler.transform(features)
                predicted_yield = self.model.predict(features_scaled)[0]
                return predicted_yield
                
            except Exception as e:
                print(f"[WARNING] Yield prediction failed: {e}")
                return self._estimate_yield_fallback(crop_type)
        else:
            return self._estimate_yield_fallback(crop_type)
    
    def _estimate_yield_fallback(self, crop_type):
        """Fallback yield estimates"""
        yield_estimates = {
            'paddy': 3500, 'rice': 4000, 'maize': 2800, 'millet': 1800,
            'buckwheat': 1200, 'potato': 18000, 'tomato': 25000,
            'onion': 15000, 'garlic': 8000, 'wheat': 2800
        }
        return yield_estimates.get(crop_type.lower(), 2000)
    
    def get_production_cost(self, crop_type, yield_kg_per_ha):
        """Calculate total production cost per hectare"""
        cost_per_kg = self.production_costs.get(crop_type.lower(), self.production_costs['default'])
        total_cost = (cost_per_kg * yield_kg_per_ha) + self.labor_cost_per_ha + self.input_cost_per_ha
        return total_cost
    
    def calculate_profit(self, crop_name, yield_kg_per_ha, area_ha=1):
        """
        Calculate profit with price source tracking
        """
        if not self.price_scraper:
            return {'success': False, 'error': "Price scraper not initialized"}
        
        # Try to get price (this will automatically use fallbacks)
        price_per_kg = self.price_scraper.get_price_per_kg(crop_name)
        
        # Determine price source
        price_source = "Unknown"
        if self.price_scraper.last_fetch_success:
            price_source = "Live from Kalimati"
        elif self.price_scraper.price_cache:
            age = datetime.now() - self.price_scraper.last_fetch_time if self.price_scraper.last_fetch_time else timedelta(days=999)
            if age.days > 0:
                price_source = f"Stored/Cached ({age.days} days old)"
            else:
                price_source = f"Stored/Cached ({age.seconds//3600} hours old)"
        else:
            price_source = "Default mock data"
        
        if price_per_kg is None:
            return {
                'success': False,
                'error': f"Price not found for crop: {crop_name}",
                'suggested_crops': self.get_similar_crops(crop_name)
            }
        
        # Calculate totals
        total_yield_kg = yield_kg_per_ha * area_ha
        total_revenue = total_yield_kg * price_per_kg
        production_cost = self.get_production_cost(crop_name, yield_kg_per_ha) * area_ha
        net_profit = total_revenue - production_cost
        profit_per_kg = price_per_kg - (production_cost / total_yield_kg) if total_yield_kg > 0 else 0
        
        return {
            'success': True,
            'crop': crop_name,
            'area_ha': area_ha,
            'yield_kg_per_ha': yield_kg_per_ha,
            'total_yield_kg': total_yield_kg,
            'price_per_kg': price_per_kg,
            'price_source': price_source,
            'price_age': self._get_price_age(),
            'total_revenue_npr': total_revenue,
            'production_cost_npr': production_cost,
            'net_profit_npr': net_profit,
            'profit_per_kg_npr': profit_per_kg,
            'roi_percent': (net_profit / production_cost) * 100 if production_cost > 0 else 0
        }
    
    def _get_price_age(self):
        """Get the age of current price data"""
        if self.price_scraper.last_fetch_time:
            age = datetime.now() - self.price_scraper.last_fetch_time
            if age.days > 0:
                return f"{age.days} days, {age.seconds//3600} hours"
            else:
                return f"{age.seconds//3600} hours, {(age.seconds%3600)//60} minutes"
        return "Unknown"
    
    def get_similar_crops(self, crop_name):
        """Get list of available crops with prices"""
        if not self.price_scraper or not self.price_scraper.price_cache:
            return []
        
        available_crops = list(self.price_scraper.price_cache.keys())
        crop_lower = crop_name.lower()
        
        similar = process.extract(
            crop_lower,
            available_crops,
            scorer=fuzz.WRatio,
            limit=5
        )
        
        return [{'name': match[0], 'similarity': match[1]} for match in similar if match[1] >= 50]


# ============================================================================
# PART 5: MAIN APPLICATION
# ============================================================================

class CropProfitAnalyzer:
    """Main application with fallback mechanisms"""
    
    def __init__(self):
        print("\n" + "="*60)
        print("CROP PROFIT ANALYZER - Kalimati Market Integration")
        print("WITH AUTOMATIC FALLBACK TO STORED PRICES")
        print("="*60)
        
        # Initialize storage first
        self.price_storage = PriceStorage()
        
        # Load model
        model, scaler, encoders, features = load_yield_model()
        
        # Initialize calculator with storage
        self.profit_calculator = ProfitCalculator(model, scaler, encoders, features, self.price_storage)
        
        # Fetch initial prices (will auto-fallback if live fails)
        print("\n[INFO] Initializing price data...")
        self.profit_calculator.price_scraper.fetch_prices()
    
    def analyze_crop(self, crop_name, district, area_ha=1, 
                     weather_data=None, soil_data=None):
        """Complete analysis for a specific crop"""
        print(f"\n" + "="*60)
        print(f"ANALYZING: {crop_name.upper()} in {district}")
        print("="*60)
        
        # Default weather and soil data
        if weather_data is None:
            weather_data = {
                'avg_temp': 25.0, 'max_temp': 32.0, 'min_temp': 18.0,
                'humidity': 65.0, 'rainfall': 1200.0,
                'solar_radiation': 6500.0, 'wind_speed': 2.5
            }
        
        if soil_data is None:
            soil_data = {'ph_value': 6.5, 'fertilizer': 100.0}
        
        # Predict yield
        predicted_yield = self.profit_calculator.predict_yield(
            district=district, crop_type=crop_name, area=area_ha,
            avg_temp=weather_data['avg_temp'], max_temp=weather_data['max_temp'],
            min_temp=weather_data['min_temp'], humidity=weather_data['humidity'],
            rainfall=weather_data['rainfall'], solar_radiation=weather_data['solar_radiation'],
            wind_speed=weather_data['wind_speed'], ph_value=soil_data['ph_value'],
            fertilizer=soil_data['fertilizer']
        )
        
        print(f"\n[YIELD PREDICTION]")
        print(f"  Expected yield: {predicted_yield:.0f} kg/ha")
        print(f"  Total for {area_ha} ha: {predicted_yield * area_ha:.0f} kg")
        
        # Calculate profit (this will use fallback prices if needed)
        profit = self.profit_calculator.calculate_profit(
            crop_name=crop_name,
            yield_kg_per_ha=predicted_yield,
            area_ha=area_ha
        )
        
        if profit['success']:
            print(f"\n[MARKET PRICE]")
            print(f"  Price Source: {profit['price_source']}")
            print(f"  Price Age: {profit['price_age']}")
            print(f"  Price: NPR {profit['price_per_kg']:.2f} per kg")
            
            print(f"\n[PROFIT ANALYSIS]")
            print(f"  Total Revenue: NPR {profit['total_revenue_npr']:,.2f}")
            print(f"  Production Cost: NPR {profit['production_cost_npr']:,.2f}")
            print(f"  NET PROFIT: NPR {profit['net_profit_npr']:,.2f}")
            print(f"  ROI: {profit['roi_percent']:.1f}%")
            
            # Save to file
            self.price_storage.save_prices(
                self.profit_calculator.price_scraper.price_cache,
                crop_name=crop_name,
                profit_data=profit
            )
            
            return profit
        else:
            print(f"\n[ERROR] {profit['error']}")
            
            if 'suggested_crops' in profit and profit['suggested_crops']:
                print(f"\n[SUGGESTED CROPS WITH PRICES]")
                for crop in profit['suggested_crops'][:5]:
                    price = self.profit_calculator.price_scraper.get_price_per_kg(crop['name'])
                    if price:
                        print(f"  - {crop['name']} (similarity: {crop['similarity']}%) - NPR {price:.2f}/kg")
            
            return profit
    
    def show_price_status(self):
        """Display current price source and age"""
        print("\n" + "="*60)
        print("PRICE DATA STATUS")
        print("="*60)
        
        if self.profit_calculator.price_scraper.last_fetch_success:
            print(f"[LIVE] Last successful fetch: {self.profit_calculator.price_scraper.last_fetch_time}")
        else:
            print(f"[CACHED/STORED] Last fetch attempt: {self.profit_calculator.price_scraper.last_fetch_time}")
        
        if self.price_storage.price_history:
            latest = self.price_storage.price_history[0]
            print(f"[STORAGE] Latest record: {latest['timestamp']}")
        
        print(f"[CACHE] {len(self.profit_calculator.price_scraper.price_cache)} crops in memory")
    
    def refresh_prices(self):
        """Force refresh prices from live source"""
        print("\n[INFO] Forcing price refresh from Kalimati...")
        prices = self.profit_calculator.price_scraper.refresh_prices()
        if prices:
            print("[OK] Prices refreshed successfully")
        return prices


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function with demonstration of fallback mechanism"""
    
    # Initialize analyzer
    analyzer = CropProfitAnalyzer()
    
    # Show current price status
    analyzer.show_price_status()
    
    # Test 1: Analyze crop (will use best available price source)
    print("\n" + "="*60)
    print("TEST 1: Analyzing Crop with Automatic Price Fallback")
    print("="*60)
    
    result = analyzer.analyze_crop(
        crop_name="paddy",
        district="Achham",
        area_ha=1
    )
    
    # Test 2: Demonstrate fallback with misspelled crop
    print("\n" + "="*60)
    print("TEST 2: Handling Misspelled Crop Name")
    print("="*60)
    
    result = analyzer.analyze_crop("paddee", "Achham")  # Misspelled
    
    # Test 3: Show price status again
    analyzer.show_price_status()
    
    print("\n" + "="*60)
    print("[SUCCESS] Crop Profit Analyzer with Fallback completed!")
    print("="*60)


if __name__ == "__main__":
    main()