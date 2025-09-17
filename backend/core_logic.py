import gpxpy
from gpxpy.gpx import GPX, GPXTrackPoint
from typing import List, Dict, Any, Optional
from models import PredictionInput, RacePlan, Segment, AthleteProfile, StravaActivity
import numpy as np
import requests
from datetime import datetime
import zipfile
import io
import csv

# --- Constantes ---
CONFIG = {
    "base_energy_cost": 1.05, "hydration_base_rate_ml_kg_hr": 10,
    "climb_threshold_m": 40, "climb_slope_threshold": 0.04,
    "significant_descent_threshold_m": 50,
    "open_meteo_api_url": "https://archive-api.open-meteo.com/v1/archive"
}

# --- Fonctions d'Analyse d'Effort et de Tri ---
def get_and_sort_strava_activities(activities: List[Dict[str, Any]]) -> List[StravaActivity]:
    sorted_activities = sorted(activities, key=lambda act: (act.get('workout_type') == 1, act.get('suffer_score', 0) or 0), reverse=True)
    return [StravaActivity(**act) for act in sorted_activities]

def _calculate_activity_effort_score(df_segments: List[Dict[str, Any]], max_hr: Optional[int]) -> float:
    if not df_segments: return 0
    score = 0
    if max_hr and any('hr' in s for s in df_segments):
        high_hr_threshold = max_hr * 0.90
        time_in_red_zone_percent = np.mean([1 for s in df_segments if s.get('hr', 0) > high_hr_threshold])
        score += time_in_red_zone_percent * 50
    if any('vam' in s for s in df_segments):
        vams = [s['vam'] for s in df_segments]
        peak_vam = np.mean(sorted(vams, reverse=True)[:10]) if len(vams) > 10 else np.mean(vams)
        score += min(1.0, peak_vam / 1500.0) * 50
    return score

# --- Classe de Traitement GPX ---
class GpxProcessor:
    def __init__(self, gpx: GPX):
        self.gpx = gpx
        self.points = []

    def process(self) -> List[Dict[str, Any]]:
        self._load_and_prepare_points()
        return self.points

    def _load_and_prepare_points(self):
        raw_points = [p[0] for p in self.gpx.walk()]
        if not raw_points or len(raw_points) < 2: return

        last_elevation = next((p.elevation for p in raw_points if p.elevation is not None), 0)
        for p in raw_points:
            if p.elevation is None: p.elevation = last_elevation
            else: last_elevation = p.elevation

        total_dist_so_far = 0.0
        self.points.append({'point': raw_points[0], 'cumulative_distance': 0})
        for i in range(1, len(raw_points)):
            dist = raw_points[i].distance_3d(raw_points[i-1]) or 0
            total_dist_so_far += dist
            self.points.append({'point': raw_points[i], 'cumulative_distance': total_dist_so_far})
    def _identify_raw_trends(self):
        trends, start_idx = [], 0
        if len(self.points) < 2: return []
        ele_diff_init = self.points[1]['point'].elevation - self.points[0]['point'].elevation
        current_trend = 'flat' if abs(ele_diff_init) < 1 else ('climb' if ele_diff_init > 0 else 'descent')
        for i in range(1, len(self.points)):
            ele_diff = self.points[i]['point'].elevation - self.points[i-1]['point'].elevation
            new_trend = 'flat' if abs(ele_diff) < 1 else ('climb' if ele_diff > 0 else 'descent')
            if new_trend != current_trend:
                trends.append({'type': current_trend, 'start_idx': start_idx, 'end_idx': i - 1})
                start_idx = i - 1
                current_trend = new_trend
        
        trends.append({'type': current_trend, 'start_idx': start_idx, 'end_idx': len(self.points) - 1})
        for trend in trends:
            start_p, end_p = self.points[trend['start_idx']], self.points[trend['end_idx']]
            trend['distance'] = end_p['cumulative_distance'] - start_p['cumulative_distance']
            trend['gain'] = sum(max(0, self.points[j]['point'].elevation - self.points[j-1]['point'].elevation) for j in range(trend['start_idx'] + 1, trend['end_idx'] + 1))
            trend['loss'] = sum(abs(min(0, self.points[j]['point'].elevation - self.points[j-1]['point'].elevation)) for j in range(trend['start_idx'] + 1, trend['end_idx'] + 1))
        return [t for t in trends if t['distance'] > 10]
    def _merge_climbs(self, trends: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not trends: return []
        merged_trends, i = [], 0
        while i < len(trends):
            current_trend = trends[i]
            if current_trend['type'] == 'climb' and i + 2 < len(trends):
                next_trend, after_next_trend = trends[i+1], trends[i+2]
                if next_trend['type'] in ['descent', 'flat'] and next_trend['loss'] < CONFIG['significant_descent_threshold_m'] and after_next_trend['type'] == 'climb':
                    current_trend['end_idx'] = after_next_trend['end_idx']
                    i += 2
                    continue
            merged_trends.append(current_trend)
            i += 1
        return merged_trends
    def find_significant_segments(self, aid_stations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.points: return []
        
        raw_trends = self._identify_raw_trends()
        merged_trends = self._merge_climbs(raw_trends)
        
        feature_points = {0, len(self.points) - 1}
        for station in aid_stations:
            idx = min(range(len(self.points)), key=lambda i: abs(self.points[i]['cumulative_distance'] - (station.get('distance', 0) * 1000)))
            feature_points.add(idx)
        for trend in merged_trends:
            if trend['type'] == 'climb' and trend['gain'] > CONFIG['climb_threshold_m']:
                feature_points.add(trend['start_idx']); feature_points.add(trend['end_idx'])
            elif trend['type'] == 'descent' and trend['loss'] > CONFIG['significant_descent_threshold_m']:
                 feature_points.add(trend['start_idx']); feature_points.add(trend['end_idx'])
        unique_indices = sorted(list(set(feature_points)))
        segments = []
        for i in range(len(unique_indices) - 1):
            start_idx, end_idx = unique_indices[i], unique_indices[i+1]
            if end_idx <= start_idx: continue
            start_p, end_p = self.points[start_idx], self.points[end_idx]
            distance = end_p['cumulative_distance'] - start_p['cumulative_distance']
            if distance < 50: continue
            elevation_gain = sum(max(0, self.points[j]['point'].elevation - self.points[j-1]['point'].elevation) for j in range(start_idx + 1, end_idx + 1))
            slope = (end_p['point'].elevation - start_p['point'].elevation) / distance if distance > 0 else 0
            segment_name = "Transition"
            if slope > CONFIG['climb_slope_threshold']: segment_name = f"Montée"
            elif slope < -CONFIG['climb_slope_threshold']: segment_name = f"Descente"
            for station in aid_stations:
                 if abs(end_p['cumulative_distance'] - (station.get('distance', 0) * 1000)) < 50:
                     segment_name = f"vers {station.get('name', 'Ravitaillement')}"
                     break

            segments.append({"name": segment_name, "distance": distance, "elevation_gain": elevation_gain, "slope": slope})
        if segments and "vers" not in segments[-1]['name'].lower(): segments[-1]['name'] = "Vers l'Arrivée"
        return segments

# --- Logique de Profilage de l'Athlète ---
class AthleteProfiler:
    def __init__(self, activities_streams: List[Dict[str, Any]], weight: float, max_hr: Optional[int] = None):
        self.streams = [s for s in activities_streams if self._is_valid_stream(s)]
        self.weight = weight
        self.max_hr = max_hr

    def _is_valid_stream(self, stream: Dict[str, Any]) -> bool:
        return 'time' in stream and 'latlng' in stream and 'altitude' in stream and 'cadence' in stream and len(stream['time']['data']) > 100
    def _get_default_profile(self) -> AthleteProfile:
        return AthleteProfile(vam=800, downhill_technique_speed=3.5, aerobic_endurance_speed=3.0, fatigue_resistance=0.95, pace_management=0.9, cadence_efficiency=85, long_distance_endurance=0.85, runner_type="Profil Standard (données insuffisantes)", radar_data={ "Puissance Montée": 50, "Technique Descente": 50, "Endurance Fondamentale": 50, "Endurance Longue": 50, "Résistance Fatigue": 50, "Gestion Allure": 50, "Efficacité Course": 50 }, analysis_level="standard", max_hr=self.max_hr, terrain_technique_score=50, performance_model_coeffs=[])
    def _build_performance_model(self, all_segments: List[Dict[str, Any]]):
        slopes = np.array([s['slope'] for s in all_segments if -0.3 < s['slope'] < 0.3])
        speeds = np.array([s['speed'] for s in all_segments if -0.3 < s['slope'] < 0.3])
        if len(slopes) < 10: self.performance_model = np.poly1d([0, 3.0]); return
        self.performance_model = np.poly1d(np.polyfit(slopes, speeds, 3))
    def calculate_profile(self) -> AthleteProfile:
        if not self.streams: return self._get_default_profile()
        all_segments, has_hr_data = [], False
        for stream in self.streams:
            df = self._stream_to_dataframe(stream)
            if df.empty: continue
            stream_has_hr = 'heartrate' in df.columns and df['heartrate'].notna().any()
            if stream_has_hr: has_hr_data = True
            all_segments.extend(self._dataframe_to_segments(df, stream_has_hr))
        if not all_segments: return self._get_default_profile()
        self._build_performance_model(all_segments)
        metrics = self._calculate_metrics(all_segments, has_hr_data)
        runner_type = self._define_runner_type(metrics['vam'], metrics['aerobic_speed'], metrics['downhill_speed'], has_hr_data)
        radar_data = self._get_radar_data(**metrics)
        return AthleteProfile(
            **metrics, runner_type=runner_type, radar_data=radar_data,
            analysis_level="detailed" if has_hr_data else "standard", max_hr=self.max_hr,
            performance_model_coeffs=self.performance_model.coeffs.tolist()
        )
    def _calculate_metrics(self, all_segments, has_hr_data):
        aerobic_speed = self.performance_model(0.0)
        downhill_speed = self.performance_model(-0.15)
        uphill_speed_at_15_percent = self.performance_model(0.15)
        vam = (uphill_speed_at_15_percent * 0.15) * 3600
        tech_score = np.mean([s['technicality'] for s in all_segments]) * 10 or 50
        cadence_efficiency = np.mean([s['cadence'] for s in all_segments if -0.02 < s['slope'] < 0.02]) or 85
        
        # Calcul de l'endurance longue distance
        short_runs_gaps = [s['gap'] for s in all_segments if s['total_distance'] < 30000 and s['gap'] > 0]
        long_runs_gaps = [s['gap'] for s in all_segments if s['total_distance'] > 50000 and s['gap'] > 0]
        avg_gap_short = np.mean(short_runs_gaps) if short_runs_gaps else 0
        avg_gap_long = np.mean(long_runs_gaps) if long_runs_gaps else 0
        long_distance_endurance = (1000 / avg_gap_long) / (1000/avg_gap_short) if avg_gap_short > 0 and avg_gap_long > 0 else 0.85

        fatigue_resistance, pace_management = 0.95, 0.9
        if has_hr_data:
            hr_segments = [s for s in all_segments if s.get('hr') and s['hr'] > 100]
            if hr_segments:
                first_half_eff = np.mean([s['gap_efficiency'] for s in hr_segments if s['progress'] < 0.5])
                second_half_eff = np.mean([s['gap_efficiency'] for s in hr_segments if s['progress'] >= 0.5])
                fatigue_resistance = (second_half_eff / first_half_eff) if first_half_eff else 0.95
                hr_variability = np.std([s['hr'] for s in hr_segments])
                pace_management = 1 - (hr_variability / np.mean([s['hr'] for s in hr_segments])) if hr_variability and np.mean([s['hr'] for s in hr_segments]) > 0 else 0.9
        return {'vam': vam, 'downhill_technique_speed': downhill_speed, 'aerobic_endurance_speed': aerobic_speed, 'fatigue_resistance': fatigue_resistance, 'pace_management': pace_management, 'cadence_efficiency': cadence_efficiency, 'terrain_technique_score': tech_score, 'long_distance_endurance': long_distance_endurance}
    def _stream_to_dataframe(self, stream: Dict[str, Any]) -> 'pd.DataFrame':
        import pandas as pd
        data_streams = {k: v['data'] for k, v in stream.items() if isinstance(v, dict) and 'data' in v and k != 'latlng'}
        latlng_stream = stream.get('latlng', {}).get('data')
        all_lengths = [len(d) for d in data_streams.values()] + ([len(latlng_stream)] if latlng_stream else [])
        if not all_lengths: return pd.DataFrame()
        max_len = max(all_lengths)
        df_data = {k: np.pad(v, (0, max_len - len(v)), 'edge') for k, v in data_streams.items()}
        if latlng_stream:
            latlng_array = np.array(latlng_stream); padded = np.pad(latlng_array, ((0, max_len - len(latlng_array)), (0, 0)), 'edge')
            df_data['latitude'], df_data['longitude'] = padded[:, 0], padded[:, 1]
        return pd.DataFrame(df_data)
    def _dataframe_to_segments(self, df: 'pd.DataFrame', has_hr: bool, seg_dur_s=30) -> list:
        segments = []
        if 'time' not in df.columns or df['time'].empty: return []
        total_time, total_distance = df['time'].iloc[-1], df['distance'].iloc[-1]
        for i in range(0, len(df) - seg_dur_s, seg_dur_s):
            sub = df.iloc[i:i+seg_dur_s]; dist = sub['distance'].iloc[-1] - sub['distance'].iloc[0]
            gain = sub['altitude'].diff().clip(lower=0).sum(); duration = sub['time'].iloc[-1] - sub['time'].iloc[0]
            if duration == 0: continue
            slope = (sub['altitude'].iloc[-1] - sub['altitude'].iloc[0]) / dist if dist > 0 else 0
            speed = dist / duration
            gap_speed = speed / (1 + (slope * 100) * 0.08 if slope > 0 else 1 + (slope * 100) * 0.03)
            pace_variability = sub['velocity_smooth'].std() if 'velocity_smooth' in sub else 0
            tortuosity = 1 - (_haversine(sub['longitude'].iloc[0], sub['latitude'].iloc[0], sub['longitude'].iloc[-1], sub['latitude'].iloc[-1]) / dist) if dist > 0 else 0
            seg = {'slope': slope, 'speed': speed, 'vam': (gain / duration) * 3600, 'progress': sub['time'].iloc[0] / total_time, 'technicality': pace_variability + (tortuosity * 10), 'cadence': sub['cadence'].mean() if 'cadence' in sub else 0, 'gap': 1000/gap_speed if gap_speed > 0 else 0, 'total_distance': total_distance}
            if has_hr: seg['hr'] = sub['heartrate'].mean(); seg['gap_efficiency'] = gap_speed / seg['hr'] if seg['hr'] > 0 else 0
            segments.append(seg)
        return segments
    def _define_runner_type(self, vam, aerobic_speed, downhill_speed, has_hr_data):
        if not has_hr_data: return "Profil Standard (données FC manquantes)";
        if vam > 1200: return "Pur Grimpeur";
        if downhill_speed * 3.6 > 15: return "Descendeur Agile";
        if aerobic_speed * 3.6 > 14: return "Coureur sur Plat Rapide";
        return "Traileur Polyvalent"
    def _get_radar_data(self, vam, downhill_technique_speed, aerobic_endurance_speed, fatigue_resistance, pace_management, terrain_technique_score, cadence_efficiency, long_distance_endurance) -> Dict[str, float]:
        return {"Puissance Montée": min(100, vam / 1500 * 100), "Technique Descente": min(100, (downhill_technique_speed * 3.6) / 20 * 100), "Endurance Fondamentale": min(100, (aerobic_endurance_speed * 3.6) / 16 * 100), "Endurance Longue": min(100, max(0, (long_distance_endurance - 0.7) / 0.25 * 100)), "Résistance Fatigue": min(100, max(0, (fatigue_resistance - 0.85) / 0.15 * 100)), "Gestion Allure": min(100, max(0, (pace_management - 0.8) / 0.15 * 100)), "Efficacité Course": min(100, (cadence_efficiency - 75) / 15 * 100)}

# --- Logique de Prédiction ---
def _get_long_distance_adjustment(distance_km: float, profile: AthleteProfile) -> float:
    """Calcule un facteur de ralentissement pour les longues distances."""
    base_slowdown = np.log1p(max(0, distance_km - 42) / 42) * 0.2 # 20% de ralentissement logarithmique au-delà du marathon
    # L'endurance de l'athlète réduit ce ralentissement
    endurance_factor = 1 - (profile.long_distance_endurance - 0.85)
    return 1 + (base_slowdown * endurance_factor)

def calculate_race_plan(params: PredictionInput, gpx: GPX, profile: AthleteProfile) -> RacePlan:
    processor = GpxProcessor(gpx)
    gpx_points = processor.process()
    segments_data = processor.find_significant_segments(params.aid_stations or [])
    total_dist_km = gpx_points[-1]['cumulative_distance'] / 1000 if gpx_points else 0
    
    performance_model = np.poly1d(profile.performance_model_coeffs)
    weather_adjustment = 0.0
    if gpx_points and params.race_date:
        start_point = gpx_points[0]['point']
        try:
            race_datetime = datetime.fromisoformat(params.race_date.replace('Z', '+00:00'))
            weather_data = get_historical_weather(start_point.latitude, start_point.longitude, race_datetime)
            weather_adjustment = calculate_weather_pace_adjustment(weather_data).get('total_adjustment_percent', 0)
        except Exception as e: print(f"Avertissement: Impossible de récupérer la météo. {e}")
    
    plan_segments: List[Segment] = []
    cumulative_time = 0.0
    long_dist_factor = _get_long_distance_adjustment(total_dist_km, profile)
    fatigue_degradation_per_hour = 1 - (profile.fatigue_resistance * 0.9 + 0.05)
    temp_factor = 1 + max(0, params.temperature - 15) * 0.1
    hydration_rate_ml_hr = CONFIG['hydration_base_rate_ml_kg_hr'] * params.weight * temp_factor

    for seg in segments_data:
        if seg['distance'] <= 0: continue
        
        base_speed = performance_model(seg['slope'])
        
        current_fatigue_factor = 1 - (fatigue_degradation_per_hour * (cumulative_time / 3600))
        weather_factor = 1 - (weather_adjustment / 100)
        final_speed = base_speed * (1 / params.tech_multiplier) * current_fatigue_factor * weather_factor / long_dist_factor
        
        time_for_segment = seg['distance'] / final_speed if final_speed > 0 else float('inf')
        
        calories = (seg['distance'] / 1000) * params.weight * CONFIG['base_energy_cost'] * (1 + max(0, seg['slope']) * 2)
        carbs_needed = (time_for_segment / 3600) * params.carb_intake
        hydration_needed = (time_for_segment / 3600) * hydration_rate_ml_hr
        cumulative_time += time_for_segment
        
        plan_segments.append(Segment(
            name=seg["name"], distance=seg['distance'], elevation_gain=seg['elevation_gain'],
            time=time_for_segment, pace=adjusted_pace, cumulative_time=cumulative_time,
            calories=calories, carbs_needed=carbs_needed, hydration_needed=hydration_needed
        ))
        
    total_dist_m = gpx_points[-1]['cumulative_distance'] if gpx_points else 0
    total_elevation = sum(s.elevation_gain for s in plan_segments)
    if params.aid_stations: cumulative_time += sum(s.get('duration', 0) for s in params.aid_stations)
    return RacePlan(plan=plan_segments, total_time=cumulative_time, total_distance=total_dist_km*1000, total_elevation=total_elevation, total_calories=sum(s.calories for s in plan_segments), total_carbs=sum(s.carbs_needed for s in plan_segments), total_hydration=sum(s.hydration_needed for s in plan_segments), weather_adjustment_percent=weather_adjustment)

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
def _haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2]); dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 6371000 * 2 * np.arcsin(np.sqrt(a))

# --- Logique de Traitement du ZIP Strava ---
def _read_activities_from_csv(csv_content: str) -> List[Dict[str, Any]]:
    activities = []
    reader = csv.DictReader(io.StringIO(csv_content))
    for row in reader:
        try:
            # On ne garde que les activités de course à pied avec un fichier GPX
            if row.get('Activity Type') == 'Run' and row.get('Filename'):
                activities.append({
                    "id": row['Activity ID'],
                    "date": row['Activity Date'],
                    "name": row['Activity Name'],
                    "type": row['Activity Type'],
                    "distance": float(row['Distance']) * 1000 if row.get('Distance') else 0,
                    "gpx_file": row['Filename']
                })
        except (ValueError, KeyError):
            continue
    return activities

def process_strava_zip(zip_file_content: io.BytesIO) -> AthleteProfile:
    """
    Traite un fichier ZIP d'export Strava pour générer un profil d'athlète.
    """
    streams = []
    weight = 70  # Poids par défaut, car non présent dans le ZIP

    with zipfile.ZipFile(zip_file_content, 'r') as z:
        # 1. Extraire les métadonnées des activités
        try:
            csv_content = z.read('activities.csv').decode('utf-8')
            activities = _read_activities_from_csv(csv_content)
        except KeyError:
            raise ValueError("Le fichier 'activities.csv' est introuvable dans le ZIP.")

        # 2. Lire les fichiers GPX correspondants pour en extraire les streams
        for activity in activities:
            gpx_path = activity['gpx_file'].replace('.gz', '') # Le nom de fichier peut inclure .gz
            if gpx_path.endswith('.gpx'):
                try:
                    gpx_content = z.read(gpx_path)
                    gpx = gpxpy.parse(gpx_content)
                    # Convertir les points GPX en un format de "stream" similaire à l'API Strava
                    # Note: C'est une simplification. Les vrais streams Strava ont plus de données.
                    stream_data = {
                        'time': {'data': [p.time_difference(gpx.tracks[0].segments[0].points[0]) for p in gpx.tracks[0].segments[0].points if p.time]},
                        'latlng': {'data': [[p.latitude, p.longitude] for p in gpx.tracks[0].segments[0].points]},
                        'altitude': {'data': [p.elevation for p in gpx.tracks[0].segments[0].points]},
                        'distance': {'data': [p.distance_from_start for p in gpx.tracks[0].segments[0].points]},
                        # Les données de cadence et FC ne sont généralement pas dans les GPX standards
                        'cadence': {'data': [85] * len(gpx.tracks[0].segments[0].points)}, # Placeholder
                        'velocity_smooth': {'data': [p.speed for p in gpx.tracks[0].segments[0].points]}
                    }
                    if all(stream_data.values()): # S'assurer que les données de base sont là
                        streams.append(stream_data)
                except (KeyError, IndexError, gpxpy.gpx.GPXXMLSyntaxException):
                    # Ignorer les fichiers GPX corrompus ou manquants
                    continue

    if not streams:
        profiler = AthleteProfiler([], weight)
        return profiler._get_default_profile()

    # 3. Générer le profil à partir des streams extraits
    profiler = AthleteProfiler(streams, weight)
    return profiler.calculate_profile()
