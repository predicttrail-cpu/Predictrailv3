import gpxpy
import zipfile
import io
import csv
import requests
import json
from typing import List, Dict, Any
from cachetools import TTLCache, cached
from fastapi import HTTPException

from ..models import StravaActivity, AthleteProfile
from .athlete_profiler import AthleteProfiler, _calculate_activity_effort_score

# --- Fonctions Utilitaires Strava ---
strava_cache = TTLCache(maxsize=100, ttl=300)

@cached(cache=strava_cache)
def fetch_strava_api(token: str, endpoint: str) -> Any:
    url = f"https://www.strava.com/api/v3/{endpoint}"
    headers = {'Authorization': f'Bearer {token}'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if e.response is not None else 500
        detail = f"Erreur API Strava ({endpoint}): {e}"
        if e.response is not None:
            try: detail = f"Erreur API Strava ({endpoint}): {e.response.json()}"
            except json.JSONDecodeError: pass
        raise HTTPException(status_code=status_code, detail=detail)

def get_and_sort_strava_activities(activities: List[Dict[str, Any]]) -> List[StravaActivity]:
    sorted_activities = sorted(activities, key=lambda act: (act.get('workout_type') == 1, act.get('suffer_score', 0) or 0), reverse=True)
    return [StravaActivity(**act) for act in sorted_activities]

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
        return profiler.get_default_profile()

    # 3. Générer le profil à partir des streams extraits
    profiler = AthleteProfiler(streams, weight)
    return profiler.calculate_profile()
