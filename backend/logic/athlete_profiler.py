from typing import List, Dict, Any, Optional
from ..models import AthleteProfile
import numpy as np

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

def _haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2]); dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 6371000 * 2 * np.arcsin(np.sqrt(a))

class AthleteProfiler:
    def __init__(self, activities_streams: List[Dict[str, Any]], weight: float, max_hr: Optional[int] = None):
        self.streams = [s for s in activities_streams if self._is_valid_stream(s)]
        self.weight = weight
        self.max_hr = max_hr

    def _is_valid_stream(self, stream: Dict[str, Any]) -> bool:
        return 'time' in stream and 'latlng' in stream and 'altitude' in stream and 'cadence' in stream and len(stream['time']['data']) > 100

    def get_default_profile(self) -> AthleteProfile:
        return AthleteProfile(vam=800, downhill_technique_speed=3.5, aerobic_endurance_speed=3.0, fatigue_resistance=0.95, pace_management=0.9, cadence_efficiency=85, long_distance_endurance=0.85, runner_type="Profil Standard (données insuffisantes)", radar_data={ "Puissance Montée": 50, "Technique Descente": 50, "Endurance Fondamentale": 50, "Endurance Longue": 50, "Résistance Fatigue": 50, "Gestion Allure": 50, "Efficacité Course": 50 }, analysis_level="standard", max_hr=self.max_hr, terrain_technique_score=50, performance_model_coeffs=[])

    def _build_performance_model(self, all_segments: List[Dict[str, Any]]):
        slopes = np.array([s['slope'] for s in all_segments if -0.3 < s['slope'] < 0.3])
        speeds = np.array([s['speed'] for s in all_segments if -0.3 < s['slope'] < 0.3])
        if len(slopes) < 10: self.performance_model = np.poly1d([0, 3.0]); return
        self.performance_model = np.poly1d(np.polyfit(slopes, speeds, 3))

    def calculate_profile(self) -> AthleteProfile:
        if not self.streams: return self.get_default_profile()
        all_segments, has_hr_data = [], False
        for stream in self.streams:
            df = self._stream_to_dataframe(stream)
            if df.empty: continue
            stream_has_hr = 'heartrate' in df.columns and df['heartrate'].notna().any()
            if stream_has_hr: has_hr_data = True
            all_segments.extend(self._dataframe_to_segments(df, stream_has_hr))
        if not all_segments: return self.get_default_profile()
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
