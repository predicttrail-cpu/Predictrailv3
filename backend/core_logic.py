import gpxpy
from gpxpy.gpx import GPX, GPXTrackPoint
from typing import List, Dict, Any, Optional
from models import PredictionInput, RacePlan, Segment, AthleteProfile, StravaActivity
import math
import numpy as np

# --- Constantes de Configuration ---
CONFIG = {
    "base_energy_cost": 1.05,
    "hydration_base_rate_ml_kg_hr": 10,
    "climb_threshold_m": 40,
    "climb_slope_threshold": 0.04,
    "significant_descent_threshold_m": 50
}

# --- Fonctions de Tri des Activités ---
def get_and_sort_strava_activities(activities: List[Dict[str, Any]]) -> List[StravaActivity]:
    sorted_activities = sorted(
        activities,
        key=lambda act: (
            act.get('workout_type') == 1,
            act.get('suffer_score', 0) or 0,
            act.get('workout_type') == 2,
            act.get('workout_type') == 3
        ),
        reverse=True
    )
    return [StravaActivity(**act) for act in sorted_activities]
    
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
        trends = []
        if len(self.points) < 2: return []
        
        start_idx = 0
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
        
        if segments and "vers" not in segments[-1]['name'].lower():
            segments[-1]['name'] = "Vers l'Arrivée"
            
        return segments

# --- Logique de Profilage de l'Athlète ---
class AthleteProfiler:
    def __init__(self, activities_streams: List[Dict[str, Any]], weight: float, max_hr: Optional[int] = None):
        self.streams = [s for s in activities_streams if self._is_valid_stream(s)]
        self.weight = weight
        self.max_hr = max_hr

    def _is_valid_stream(self, stream: Dict[str, Any]) -> bool:
        return 'time' in stream and 'latlng' in stream and 'altitude' in stream and len(stream['time']['data']) > 100

    def _get_default_profile(self) -> AthleteProfile:
        return AthleteProfile(
            vam=800, downhill_technique_speed=3.5, aerobic_endurance_speed=3.0,
            fatigue_resistance=0.95, pace_management=0.9,
            runner_type="Profil Standard (données insuffisantes)",
            radar_data={ "Puissance Montée": 50, "Technique Descente": 50, "Endurance Fondamentale": 50, "Résistance Fatigue": 50, "Gestion Allure": 50 },
            analysis_level="standard", max_hr=self.max_hr
        )

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

        vam, downhill_speed, aerobic_speed, fatigue_resistance, pace_management = self._calculate_metrics(all_segments, has_hr_data)
        runner_type = self._define_runner_type(vam, aerobic_speed, downhill_speed, has_hr_data)
        radar_data = self._get_radar_data(vam, downhill_speed, aerobic_speed, fatigue_resistance, pace_management)
        
        return AthleteProfile(
            vam=vam, downhill_technique_speed=downhill_speed, aerobic_endurance_speed=aerobic_speed,
            fatigue_resistance=fatigue_resistance, pace_management=pace_management,
            runner_type=runner_type, radar_data=radar_data,
            analysis_level="detailed" if has_hr_data else "standard", max_hr=self.max_hr
        )

    def _calculate_metrics(self, all_segments, has_hr_data):
        vam = np.mean([s['vam'] for s in all_segments if s['vam'] > 300 and s['vam'] < 2500]) or 800
        downhill_speed = np.mean([s['speed'] for s in all_segments if s['slope'] < -0.10 and s['speed'] > 1]) or 3.5
        aerobic_speed = np.mean([s['speed'] for s in all_segments if -0.02 < s['slope'] < 0.02]) or 3.0
        
        fatigue_resistance, pace_management = 0.95, 0.9
        if has_hr_data:
            hr_segments = [s for s in all_segments if s.get('hr') and s['hr'] > 100]
            if hr_segments:
                first_half_eff = np.mean([s['speed']/s['hr'] for s in hr_segments if s['progress'] < 0.5])
                second_half_eff = np.mean([s['speed']/s['hr'] for s in hr_segments if s['progress'] >= 0.5])
                fatigue_resistance = (second_half_eff / first_half_eff) if first_half_eff else 0.95
                hr_variability = np.std([s['hr'] for s in hr_segments])
                pace_management = 1 - (hr_variability / np.mean([s['hr'] for s in hr_segments])) if hr_variability and np.mean([s['hr'] for s in hr_segments]) > 0 else 0.9

        return vam, downhill_speed, aerobic_speed, fatigue_resistance, pace_management

    def _stream_to_dataframe(self, stream: Dict[str, Any]) -> 'pd.DataFrame':
        import pandas as pd
        data_streams = {k: v['data'] for k, v in stream.items() if isinstance(v, dict) and 'data' in v and k != 'latlng'}
        latlng_stream = stream.get('latlng', {}).get('data')
        all_lengths = [len(d) for d in data_streams.values()] + ([len(latlng_stream)] if latlng_stream else [])
        if not all_lengths: return pd.DataFrame()
        max_len = max(all_lengths)
        df_data = {k: np.pad(v, (0, max_len - len(v)), 'edge') for k, v in data_streams.items()}
        if latlng_stream:
            latlng_array = np.array(latlng_stream)
            padded = np.pad(latlng_array, ((0, max_len - len(latlng_array)), (0, 0)), 'edge')
            df_data['latitude'], df_data['longitude'] = padded[:, 0], padded[:, 1]
        return pd.DataFrame(df_data)
    
    def _dataframe_to_segments(self, df: 'pd.DataFrame', has_hr: bool, seg_dur_s=30) -> list:
        segments = []
        if 'time' not in df.columns or df['time'].empty: return []
        total_time = df['time'].iloc[-1]
        for i in range(0, len(df) - seg_dur_s, seg_dur_s):
            sub = df.iloc[i:i+seg_dur_s]
            dist = sub['distance'].iloc[-1] - sub['distance'].iloc[0]
            gain = sub['altitude'].diff().clip(lower=0).sum()
            duration = sub['time'].iloc[-1] - sub['time'].iloc[0]
            if duration == 0: continue
            seg = {'slope': (sub['altitude'].iloc[-1] - sub['altitude'].iloc[0]) / dist if dist > 0 else 0,
                   'speed': dist / duration, 'vam': (gain / duration) * 3600,
                   'progress': sub['time'].iloc[0] / total_time}
            if has_hr: seg['hr'] = sub['heartrate'].mean()
            segments.append(seg)
        return segments

    def _define_runner_type(self, vam, aerobic_speed, downhill_speed, has_hr_data):
        if not has_hr_data: return "Profil Standard (données FC manquantes)"
        if vam > 1200: return "Pur Grimpeur"
        if downhill_speed * 3.6 > 15: return "Descendeur Agile"
        if aerobic_speed * 3.6 > 14: return "Coureur sur Plat Rapide"
        return "Traileur Polyvalent"

    def _get_radar_data(self, vam, downhill, aerobic, fatigue, pacing) -> Dict[str, float]:
        return {
            "Puissance Montée": min(100, vam / 1500 * 100),
            "Technique Descente": min(100, (downhill * 3.6) / 20 * 100),
            "Endurance Fondamentale": min(100, (aerobic * 3.6) / 16 * 100),
            "Résistance Fatigue": min(100, max(0, (fatigue - 0.85) / 0.15 * 100)),
            "Gestion Allure": min(100, max(0, (pacing - 0.8) / 0.15 * 100))
        }

# --- Logique de Prédiction ---
def calculate_race_plan(params: PredictionInput, gpx: GPX, profile: AthleteProfile) -> RacePlan:
    processor = GpxProcessor(gpx)
    gpx_points = processor.process()
    segments_data = processor.find_significant_segments(params.aid_stations or [])

    plan_segments: List[Segment] = []
    cumulative_time = 0.0
    
    # Utilisation du profil de puissance pour un calcul plus stable
    base_pace_on_flat_s_per_km = 1000 / profile.aerobic_endurance_speed if profile.aerobic_endurance_speed > 0 else 480

    # Facteurs de nutrition
    temp_factor = 1 + max(0, params.temperature - 15) * 0.1
    hydration_rate_ml_hr = CONFIG['hydration_base_rate_ml_kg_hr'] * params.weight * temp_factor

    for seg in segments_data:
        if seg['distance'] <= 0: continue
        
        slope = seg['slope']
        
        # Modèle de coût simple et robuste pour ajuster l'allure de base
        # 8% de pénalité par % de pente positive, 3% de bonus par % de pente négative
        cost_factor = 1 + (slope * 100) * 0.08 
        if slope < -0.02: 
            cost_factor = 1 + (slope * 100) * 0.03
        
        # L'allure est ajustée par la pente et la technicité
        adjusted_pace = base_pace_on_flat_s_per_km * cost_factor * params.tech_multiplier
        
        time_for_segment = (seg['distance'] / 1000) * adjusted_pace
        
        # Calculs de nutrition
        calories = (seg['distance'] / 1000) * params.weight * CONFIG['base_energy_cost'] * cost_factor
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
    
    if params.aid_stations:
        cumulative_time += sum(s.get('duration', 0) for s in params.aid_stations)

    return RacePlan(
        plan=plan_segments, total_time=cumulative_time,
        total_distance=total_dist_m, total_elevation=total_elevation,
        total_calories=sum(s.calories for s in plan_segments),
        total_carbs=sum(s.carbs_needed for s in plan_segments),
        total_hydration=sum(s.hydration_needed for s in plan_segments)
    )

