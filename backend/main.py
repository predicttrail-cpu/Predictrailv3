from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import gpxpy

# Importer les modèles de données et la logique de calcul
from models import PredictionInput, RacePlan
from core_logic import calculate_race_plan

app = FastAPI(
    title="PredictTrail API",
    description="API pour le calcul de plans de course de trail.",
    version="1.1.0"
)

# Configuration CORS pour autoriser les requêtes depuis le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Pour le développement. À restreindre en production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    """ Point de terminaison racine pour vérifier que le serveur est en ligne. """
    return {"status": "PredictTrail API is running."}

@app.post("/predict", response_model=RacePlan)
async def predict_race_plan(
    gpx_file: UploadFile = File(..., description="Le fichier GPX du parcours."),
    athlete_data_str: str = Form(..., description="Les données de l'athlète et de la course au format JSON.")
):
    """
    Point de terminaison principal pour la prédiction.
    Reçoit un fichier GPX et un formulaire contenant les données de l'athlète.
    """
    if not gpx_file.filename.endswith('.gpx'):
        raise HTTPException(status_code=400, detail="Fichier invalide. Veuillez uploader un fichier .gpx.")

    try:
        # On parse la chaîne JSON pour reconstruire l'objet PredictionInput
        athlete_data_dict = json.loads(athlete_data_str)
        athlete_data = PredictionInput(**athlete_data_dict)

        # Lire et analyser le fichier GPX
        gpx_content = await gpx_file.read()
        gpx = gpxpy.parse(gpx_content)

        # Appeler la logique de calcul principale
        race_plan_result = calculate_race_plan(athlete_data, gpx)

        return race_plan_result

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Erreur dans le format des données de l'athlète (JSON invalide).")
    except gpxpy.gpx.GPXXMLSyntaxException:
        raise HTTPException(status_code=400, detail="Erreur de syntaxe dans le fichier GPX. Le fichier est peut-être corrompu.")
    except ValueError as e:
        # Erreurs métier levées par notre logique de calcul
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Gérer les autres erreurs potentielles
        print(f"An unexpected error occurred: {e}") # Log pour le débogage serveur
        raise HTTPException(status_code=500, detail=f"Une erreur interne inattendue est survenue: {str(e)}")

