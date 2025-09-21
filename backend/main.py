import os
import requests
import gpxpy
import json
import io
from fastapi import FastAPI, Form, HTTPException, UploadFile, File, Request, Header
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from typing import List

from models import PredictionInput, RacePlan, StravaActivity, AthleteProfile
from logic.race_planner import calculate_race_plan
from logic.strava_utils import get_and_sort_strava_activities, process_strava_zip, fetch_strava_api
from logic.gpx_processor import GpxProcessor
from logic.athlete_profiler import AthleteProfiler, get_athlete_profile_logic

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
        return await get_athlete_profile_logic(token, manual_ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    authorization: str = Header(None),
    athlete_data_str: str = Form(...),
    gpx_file: UploadFile = File(...)
):
    try:
        params = PredictionInput(**json.loads(athlete_data_str))
        
        if authorization:
            token = authorization.split(" ")[1]
            athlete_data = fetch_strava_api(token, "athlete")
            params.weight = athlete_data.get('weight') or params.weight
            manual_ids = params.manual_activity_ids or []
            profile = await get_athlete_profile_logic(token, manual_ids)
        else:
            profile = AthleteProfile(**params.profile) if hasattr(params, 'profile') else None
            if not profile:
                 profiler = AthleteProfiler([], params.weight)
                 profile = profiler.get_default_profile()

        gpx = gpxpy.parse(await gpx_file.read())
        race_plan = calculate_race_plan(params, gpx, profile)
        return race_plan
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {e}")

app.mount("/", StaticFiles(directory="static", html=True), name="static")
