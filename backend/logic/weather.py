import requests
from datetime import datetime

# --- Constantes ---
CONFIG = {
    "open_meteo_api_url": "https://archive-api.open-meteo.com/v1/archive"
}

# --- Fonctions Météo et Géodésiques ---
def get_historical_weather(lat: float, lon: float, start_date: datetime) -> dict:
    params = {'latitude': lat, 'longitude': lon, 'start_date': start_date.strftime('%Y-%m-%d'), 'end_date': start_date.strftime('%Y-%m-%d'), 'hourly': 'temperature_2m,relative_humidity_2m'}
    try:
        response = requests.get(CONFIG['open_meteo_api_url'], params=params); response.raise_for_status(); data = response.json(); hour_index = start_date.hour
        return {'temperature_celsius': data['hourly']['temperature_2m'][hour_index], 'relative_humidity_percent': data['hourly']['relative_humidity_2m'][hour_index]}
    except Exception as e: print(f"Erreur météo: {e}"); return {}

def calculate_weather_pace_adjustment(weather_data: dict) -> dict:
    temp_c = weather_data.get('temperature_celsius', 15); humidity = weather_data.get('relative_humidity_percent', 50); temp_f = (temp_c * 9/5) + 32
    adjustment_percent = max(0, (temp_f - 60) / 5)
    return {'total_adjustment_percent': adjustment_percent}
