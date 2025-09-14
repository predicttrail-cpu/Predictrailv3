from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
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
        response = requests.post(token_exchange_url, data=payload)
        response.raise_for_status() # Lève une exception si la requête échoue
        token_data = response.json()
        
        # Pour l'instant, on affiche juste les données.
        # Plus tard, on les sauvegardera dans une base de données liée à l'utilisateur.
        print("Token reçu:", token_data)
        athlete_name = token_data.get("athlete", {}).get("firstname", "l'athlète")
        
        # Idéalement, on redirigerait l'utilisateur vers le frontend avec un message de succès
        # ou en stockant le token dans un cookie sécurisé.
        return {"message": f"Authentification réussie pour {athlete_name}!", "data": token_data}

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de l'échange du token avec Strava: {e}")


# --- Route de prédiction (inchangée pour l'instant) ---
@app.post("/predict", response_model=RacePlan)
async def predict_race_plan(
    gpx_file: UploadFile = File(...),
    athlete_data_str: str = Form(...)
):
    # ... (le code de cette fonction reste identique)
    if not gpx_file.filename or not gpx_file.filename.endswith('.gpx'):
        raise HTTPException(status_code=400, detail="Fichier invalide. Veuillez uploader un fichier .gpx.")
    try:
        athlete_data_dict = json.loads(athlete_data_str)
        athlete_data = PredictionInput(**athlete_data_dict)
        gpx_content = await gpx_file.read()
        gpx = gpxpy.parse(gpx_content)
        race_plan_result = calculate_race_plan(athlete_data, gpx)
        return race_plan_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    