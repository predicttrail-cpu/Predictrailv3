import os
import requests
import gpxpy
import json
import io
from fastapi import FastAPI, Form, HTTPException, UploadFile, File, Request, Header
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from typing import List, Any

from backend.models import PredictionInput, RacePlan, StravaActivity, AthleteProfile
from backend.logic.race_planner import calculate_race_plan
from backend.logic.strava_utils import get_and_sort_strava_activities, process_strava_zip, fetch_strava_api
from backend.logic.gpx_processor import GpxProcessor
from backend.logic.athlete_profiler import AthleteProfiler, _calculate_activity_effort_score

load_dotenv()
STRAVA_CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")
STRAVA_CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")

app = FastAPI(title="Kairn API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- ROUTES API ---
@app.get("/api/config")
def get_config(): return {"strava_client_id": STRAVA_CLIENT_ID}

@app.post("/api/auth/strava/token")
async def strava_token(code: str = Form(...)):
    token_url = "https://www.strava.com/api/v3/oauth/token"
    payload = { "client_id": STRAVA_CLIENT_ID, "client_secret": STRAVA_CLIENT_SECRET, "code": code, "grant_type": "authorization_code" }
    try:
        response = requests.post(token_url, data=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Erreur d'échange de token: {e.response.json()}")

@app.get("/api/strava/activities", response_model=List[StravaActivity])
async def get_user_activities(authorization: str = Header(...)):
    token = authorization.split(" ")[1]
    activities = fetch_strava_api(token, "athlete/activities?per_page=100")
    return get_and_sort_strava_activities(activities)

@app.post("/api/athlete/profile", response_model=AthleteProfile)
async def get_athlete_profile(request: Request, authorization: str = Header(...)):
    token = authorization.split(" ")[1]
    manual_ids = json.loads(request.headers.get('x-manual-activity-ids', '[]'))
    
    try:
        athlete_data = fetch_strava_api(token, "athlete")
        max_hr = athlete_data.get('max_hr')
        activity_ids_to_fetch = manual_ids

        if not manual_ids:
            all_activities = fetch_strava_api(token, "athlete/activities?per_page=50")
            races = [act for act in all_activities if act.get('workout_type') == 1 and act.get('distance',0) > 5000]
            if races:
                activity_ids_to_fetch = [r['id'] for r in races]
            else:
                efforts = []
                for act in all_activities:
                    if act.get('type') == 'Run' and act.get('distance', 0) > 5000:
                        try:
                            stream = fetch_strava_api(token, f"activities/{act['id']}/streams?keys=time,altitude,heartrate,velocity_smooth,distance,cadence&key_by_type=true")
                            profiler = AthleteProfiler([stream], athlete_data.get('weight', 70), max_hr)
                            df = profiler._stream_to_dataframe(stream)
                            df_segments = profiler._dataframe_to_segments(df, 'heartrate' in df.columns)
                            score = _calculate_activity_effort_score(df_segments, max_hr)
                            efforts.append({'id': act['id'], 'score': score})
                        except Exception: continue
                top_efforts = sorted(efforts, key=lambda x: x['score'], reverse=True)[:5]
                activity_ids_to_fetch = [e['id'] for e in top_efforts]

        streams = []
        for act_id in activity_ids_to_fetch:
            try: streams.append(fetch_strava_api(token, f"activities/{act_id}/streams?keys=time,latlng,distance,altitude,heartrate,velocity_smooth,cadence&key_by_type=true"))
            except HTTPException as e:
                if e.status_code == 404: print(f"Avertissement: Activité {act_id} ignorée (404).")
                else: raise

        athlete_data = fetch_strava_api(token, "athlete")
        profiler = AthleteProfiler(streams, athlete_data.get('weight', 70), athlete_data.get('max_hr'))
        return profiler.calculate_profile()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de profilage: {e}")

@app.post("/api/gpx/parse")
async def parse_gpx_file(gpx_file: UploadFile = File(...)):
    try:
        gpx = gpxpy.parse(await gpx_file.read())
        processor = GpxProcessor(gpx)
        points = processor.process()
        if not points:
            raise HTTPException(status_code=400, detail="Aucun point valide trouvé dans le fichier GPX.")
        response_points = [{"lat": p['point'].latitude, "lon": p['point'].longitude} for p in points]
        return {"points": response_points}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur d'analyse GPX: {e}")

@app.post("/api/zip/process", response_model=AthleteProfile)
async def process_zip_file(zip_file: UploadFile = File(...)):
    try:
        zip_content = await zip_file.read()
        profile = process_strava_zip(io.BytesIO(zip_content))
        return profile
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de traitement du ZIP: {e}")

@app.post("/api/predict", response_model=RacePlan)
async def predict_race_plan_endpoint(
    request: Request,
    authorization: str = Header(None),
    athlete_data_str: str = Form(...),
    gpx_file: UploadFile = File(...)
):
    try:
        params = PredictionInput(**json.loads(athlete_data_str))
        
        # Si un token Strava est fourni, on l'utilise pour enrichir les données
        if authorization:
            token = authorization.split(" ")[1]
            athlete_data = fetch_strava_api(token, "athlete")
            params.weight = athlete_data.get('weight') or params.weight
            profile_request = Request(scope={"type": "http", "headers": request.headers.raw})
            profile = await get_athlete_profile(profile_request, authorization)
        else:
            # Sinon, on utilise un profil généré (par ex. depuis le ZIP) ou un profil par défaut
            # Pour l'instant, on se base sur ce qui est envoyé, en attendant une gestion de session
            profile = AthleteProfile(**params.profile) if hasattr(params, 'profile') else None
            if not profile:
                 # Fallback sur un profil par défaut si rien n'est fourni
                 profiler = AthleteProfiler([], params.weight)
                 profile = profiler.get_default_profile()


        gpx = gpxpy.parse(await gpx_file.read())
        race_plan = calculate_race_plan(params, gpx, profile)
        return race_plan
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {e}")

app.mount("/", StaticFiles(directory="static", html=True), name="static")
