import gpxpy
from gpxpy.gpx import GPX
from typing import List, Dict, Any

# --- Constantes ---
CONFIG = {
    "base_energy_cost": 1.05, "hydration_base_rate_ml_kg_hr": 10,
    "climb_threshold_m": 40, "climb_slope_threshold": 0.04,
    "significant_descent_threshold_m": 50,
    "open_meteo_api_url": "https://archive-api.open-meteo.com/v1/archive"
}

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
