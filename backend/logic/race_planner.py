import numpy as np
from gpxpy.gpx import GPX
from typing import List
from datetime import datetime

from ..models import PredictionInput, RacePlan, Segment, AthleteProfile
from .gpx_processor import GpxProcessor, CONFIG
from .weather import get_historical_weather, calculate_weather_pace_adjustment

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

        pace = time_for_segment / (seg['distance'] / 1000) if seg['distance'] > 0 else 0

        plan_segments.append(Segment(
            name=seg["name"], distance=seg['distance'], elevation_gain=seg['elevation_gain'],
            time=time_for_segment, pace=pace, cumulative_time=cumulative_time,
            calories=calories, carbs_needed=carbs_needed, hydration_needed=hydration_needed
        ))

    total_dist_m = gpx_points[-1]['cumulative_distance'] if gpx_points else 0
    total_elevation = sum(s.elevation_gain for s in plan_segments)
    if params.aid_stations: cumulative_time += sum(s.get('duration', 0) for s in params.aid_stations)
    return RacePlan(plan=plan_segments, total_time=cumulative_time, total_distance=total_dist_km*1000, total_elevation=total_elevation, total_calories=sum(s.calories for s in plan_segments), total_carbs=sum(s.carbs_needed for s in plan_segments), total_hydration=sum(s.hydration_needed for s in plan_segments), weather_adjustment_percent=weather_adjustment)
