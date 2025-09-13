import math
from typing import List
from models import PredictionInput, RacePlan, Segment, ReferencePerformance
import gpxpy
import gpxpy.gpx

CONFIG = {
    "baseEnergyCost": 1.0,      # kcal/kg/km
    "minettiUphillCost": 8.0,
    "minettiDownhillCost": 2.0,
}

# --- Fonctions Utilitaires ---

def parse_time_to_seconds(time_str: str) -> int:
    try:
        parts = list(map(int, time_str.split(':')))
        if len(parts) == 3: return parts[0] * 3600 + parts[1] * 60 + parts[2]
        if len(parts) == 2: return parts[0] * 60 + parts[1]
        if len(parts) == 1: return parts[0]
        return 0
    except (ValueError, IndexError):
        return 0

def get_minetti_cost_factor(slope: float) -> float:
    if slope >= 0: return 1 + CONFIG["minettiUphillCost"] * slope
    return 1 - CONFIG["minettiDownhillCost"] * slope

def calculate_hourly_hydration(weight: float, temperature: int) -> float:
    rate_per_kg = 8
    if 10 <= temperature < 15: rate_per_kg = 10
    elif 15 <= temperature < 20: rate_per_kg = 12
    elif 20 <= temperature < 25: rate_per_kg = 15
    elif temperature >= 25: rate_per_kg = 18
    return weight * rate_per_kg

# --- Logique de Calcul de l'Allure de Base ---

def calculate_equivalent_flat_pace(ref: dict, ref_tech_multiplier: float) -> float:
    if ref['distance'] <= 0: return 0
    total_elev = ref['elevGain'] + ref['elevLoss']
    if total_elev == 0: return ref['time'] / (ref['distance'] * ref_tech_multiplier)
    
    dist_up = ref['distance'] * (ref['elevGain'] / total_elev)
    dist_down = ref['distance'] - dist_up
    slope_up = ref['elevGain'] / dist_up if dist_up > 0 else 0
    slope_down = -ref['elevLoss'] / dist_down if dist_down > 0 else 0
    
    cost_factor_up = get_minetti_cost_factor(slope_up)
    cost_factor_down = get_minetti_cost_factor(slope_down)
    avg_minetti_factor = ((cost_factor_up * dist_up) + (cost_factor_down * dist_down)) / ref['distance']
    
    return ref['time'] / (ref['distance'] * avg_minetti_factor * ref_tech_multiplier)

def get_average_base_pace(references: List[ReferencePerformance]) -> float:
    base_paces = []
    for r in references:
        ref = {
            "distance": r.distance * 1000,
            "time": parse_time_to_seconds(r.time),
            "elevGain": r.elevGain,
            "elevLoss": r.elevLoss,
        }
        ref_tech_multiplier = 1 + (r.tech - 1) * 0.075
        if ref["distance"] > 0 and ref["time"] > 0:
            pace = calculate_equivalent_flat_pace(ref, ref_tech_multiplier)
            base_paces.append(pace)
            
    if not base_paces:
        raise ValueError("Aucune performance de référence valide n'a été fournie.")
    return sum(base_paces) / len(base_paces)

# --- Logique de Traitement du GPX ---

def process_gpx_points(gpx: gpxpy.gpx.GPX) -> List[dict]:
    all_points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                if point.elevation is not None:
                    all_points.append(point)
    
    if not all_points:
        raise ValueError("Aucun point avec des données d'altitude n'a été trouvé dans le fichier GPX.")

    processed_points = []
    cumulative_distance = 0
    for i, p in enumerate(all_points):
        if i > 0:
            dist_from_prev = p.distance_2d(all_points[i-1]) or 0
            cumulative_distance += dist_from_prev
        processed_points.append({
            'lat': p.latitude, 'lon': p.longitude, 'ele': p.elevation, 
            'cumulativeDistance': cumulative_distance
        })
    return processed_points

def process_segment(segment_points: List[dict]) -> dict:
    if len(segment_points) < 2: return {"distance": 0, "elevationGain": 0, "slope": 0}
    start_point = segment_points[0]
    end_point = segment_points[-1]
    distance = end_point['cumulativeDistance'] - start_point['cumulativeDistance']
    elevation_gain = sum(
        segment_points[i]['ele'] - segment_points[i-1]['ele']
        for i in range(1, len(segment_points))
        if segment_points[i]['ele'] > segment_points[i-1]['ele']
    )
    slope = (end_point['ele'] - start_point['ele']) / distance if distance > 0 else 0
    return {"distance": distance, "elevationGain": elevation_gain, "slope": slope}

def create_segments(points: List[dict], params: PredictionInput) -> List[dict]:
    if params.aidStations:
        segments = []
        last_idx = 0
        for station in params.aidStations:
            next_idx = next((i for i, p in enumerate(points) if p['cumulativeDistance'] >= station.cumulativeDistance), len(points) - 1)
            if next_idx > last_idx:
                segment_data = process_segment(points[last_idx : next_idx + 1])
                segments.append({**segment_data, "name": f"Vers {station.name}"})
                last_idx = next_idx
        if last_idx < len(points) - 1:
            segment_data = process_segment(points[last_idx:])
            segments.append({**segment_data, "name": "Vers l'Arrivée"})
        return segments
    else:
        segments, last_idx = [], 0
        seg_len_m = params.splitDistance * 1000
        for i in range(1, len(points)):
            dist_from_last = points[i]['cumulativeDistance'] - points[last_idx]['cumulativeDistance']
            if dist_from_last >= seg_len_m or i == len(points) - 1:
                segments.append(process_segment(points[last_idx : i + 1]))
                last_idx = i
        return segments

# --- Fonction Principale ---

def calculate_race_plan(params: PredictionInput, gpx: gpxpy.gpx.GPX) -> RacePlan:
    base_pace_sec_per_m = get_average_base_pace(params.references)
    gpx_points = process_gpx_points(gpx)
    segments_data = create_segments(gpx_points, params)
    
    hourly_hydration = calculate_hourly_hydration(params.weight, params.temperature)
    energy_cost_multiplier = 0.95 if params.gender == 'femme' else 1.0
    
    rough_total_time = sum(
        s['distance'] * base_pace_sec_per_m * get_minetti_cost_factor(s['slope']) * params.techMultiplier
        for s in segments_data
    )
    
    plan_segments: List[Segment] = []
    cumulative_time = 0
    
    for i, seg in enumerate(segments_data):
        if seg['distance'] <= 0: continue
        
        minetti_factor = get_minetti_cost_factor(seg['slope'])
        fatigue_multiplier = 1 + (params.fatigueFactor * (cumulative_time / (rough_total_time or 1)))
        
        t_i = seg['distance'] * base_pace_sec_per_m * minetti_factor * params.techMultiplier * fatigue_multiplier
        calories = (seg['distance'] / 1000) * params.weight * CONFIG['baseEnergyCost'] * energy_cost_multiplier * minetti_factor
        carbs = (t_i / 3600) * params.carbIntake
        hydration = (t_i / 3600) * hourly_hydration
        
        cumulative_time += t_i
        
        plan_segments.append(Segment(
            name=seg.get("name") or f"Segment {i+1}",
            distance=seg['distance'],
            elevationGain=seg['elevationGain'],
            splitTime=t_i,
            pace=t_i / (seg['distance'] / 1000) if seg['distance'] > 0 else 0,
            cumulativeTime=cumulative_time,
            calories=calories,
            carbsNeeded=carbs,
            hydrationNeeded=hydration,
        ))

    if not plan_segments:
        raise ValueError("Aucun segment de course n'a pu être généré. Le GPX est peut-être trop court ou les paramètres sont invalides.")

    total_dist_km = gpx_points[-1]['cumulativeDistance'] / 1000 if gpx_points else 0
    total_elevation = sum(s.elevationGain for s in plan_segments)
    total_calories = sum(s.calories for s in plan_segments)

    return RacePlan(
        plan=plan_segments,
        totalTime=cumulative_time,
        totalDistance=total_dist_km,
        totalElevation=total_elevation,
        totalCalories=total_calories,
        hourlyHydration=hourly_hydration
    )

