import os
import requests
import gpxpy
import json
from fastapi import FastAPI, Form, HTTPException, UploadFile, File, Request, Header
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
import json
import requests # Import de la nouvelle librairie
    
from models import PredictionInput, RacePlan
from core_logic import calculate_race_plan
import gpxpy
    
# --- CONFIGURATION STRAVA ---
    # ATTENTION: Ne jamais mettre ces clés en dur dans le code sur GitHub!
    # Pour le développement, vous pouvez les mettre ici temporairement.
    # Pour la production, utilisez des variables d'environnement.
STRAVA_CLIENT_ID =   146343   # Remplacez par votre Client ID
STRAVA_CLIENT_SECRET = "6f240bf672c156cb074ec22e1f56ed8bc50f77ce"  # Remplacez par votre Client Secret
STRAVA_REDIRECT_URI = "http://localhost:8000/auth/strava/callback"
    
app = FastAPI(
        title="PredicTrail API",
        description="Une API pour calculer des plans de course et s'authentifier avec Strava.",
        version="1.1.0"
    )
    
app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
@app.get("/")
def read_root():
        return {"status": "API is running."}
    
# --- NOUVELLE ROUTE : Redirection vers Strava pour la connexion ---
@app.get("/login/strava")
def login_strava():
    """
    Redirige l'utilisateur vers la page d'autorisation de Strava.
    """
    strava_auth_url = (
        f"https://www.strava.com/oauth/authorize?"
        f"client_id={STRAVA_CLIENT_ID}&"
        f"redirect_uri={STRAVA_REDIRECT_URI}&"
        f"response_type=code&"
        f"approval_prompt=force&"
        f"scope=read,activity:read" # On demande la permission de lire les infos et activités
    )
    return RedirectResponse(url=strava_auth_url)

# --- NOUVELLE ROUTE : Le "Callback" que Strava appelle après autorisation ---
@app.get("/auth/strava/callback")
def auth_strava_callback(code: str = Query(...)):
    """
    Strava redirige ici après que l'utilisateur a autorisé l'application.
    On échange le 'code' temporaire contre un 'access_token' permanent.
    """
    token_exchange_url = "https://www.strava.com/oauth/token"
    payload = {
        "client_id": STRAVA_CLIENT_ID,
        "client_secret": STRAVA_CLIENT_SECRET,
        "code": code,
        "grant_type": "authorization_code"
    }

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
        activity_ids_to_fetch = manual_ids
        if not manual_ids:
            all_activities = fetch_strava_api(token, "athlete/activities?per_page=100")
            races = [act for act in all_activities if act.get('workout_type') == 1]
            if len(races) < 3:
                suffer_score_activities = sorted([act for act in all_activities if (act.get('suffer_score') or 0) > 0], key=lambda x: x.get('suffer_score', 0), reverse=True)
                relevant_activities = suffer_score_activities[:5]
            else:
                relevant_activities = races
            activity_ids_to_fetch = [act['id'] for act in relevant_activities]

        streams = []
        for act_id in activity_ids_to_fetch:
            try: streams.append(fetch_strava_api(token, f"activities/{act_id}/streams?keys=time,latlng,distance,altitude,heartrate&key_by_type=true"))
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

@app.post("/api/predict", response_model=RacePlan)
async def predict_race_plan_endpoint(
    request: Request,
    authorization: str = Header(...),
    athlete_data_str: str = Form(...),
    gpx_file: UploadFile = File(...)
):
    token = authorization.split(" ")[1]
    try:
        params = PredictionInput(**json.loads(athlete_data_str))
        
        athlete_data = fetch_strava_api(token, "athlete")
        params.weight = athlete_data.get('weight') or params.weight
        
        # Le profil est généré une seule fois au début de la prédiction
        profile_request = Request(scope={"type": "http", "headers": request.headers.raw})
        profile = await get_athlete_profile(profile_request, authorization)
        
        gpx = gpxpy.parse(await gpx_file.read())
        
        # La prédiction n'a pas besoin des activités, seulement du profil généré
        race_plan = calculate_race_plan(params, gpx, profile)
        return race_plan
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {e}")

app.mount("/", StaticFiles(directory="static", html=True), name="static")

