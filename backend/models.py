from pydantic import BaseModel
from typing import List, Optional

# --- Modèles pour les entrées de l'API ---

class ReferencePerformance(BaseModel):
    """ Définit la structure d'une performance de référence. """
    distance: float  # en km
    time: str        # format "hh:mm:ss"
    elevGain: int    # en mètres
    elevLoss: int    # en mètres
    tech: float      # de 1 à 5

class AidStation(BaseModel):
    """ Définit un point de ravitaillement personnalisé. """
    name: str
    cumulativeDistance: float # en mètres

class PredictionInput(BaseModel):
    """ Modèle principal pour les données envoyées à l'API. """
    weight: float
    gender: str # "homme" ou "femme"
    references: List[ReferencePerformance]
    
    # Paramètres de la course cible
    techMultiplier: float # Note: Ceci est le multiplicateur (ex: 1.15), pas la note (ex: 3/5)
    fatigueFactor: float  # Ex: 0.15 pour 15%
    carbIntake: int       # en g/h
    temperature: int      # en °C

    # Segmentation
    splitDistance: Optional[float] = None # en km, utilisé si pas de ravitos
    aidStations: Optional[List[AidStation]] = None


# --- Modèles pour les sorties de l'API ---

class Segment(BaseModel):
    """ Définit la structure d'un segment du plan de course. """
    name: str
    distance: float        # en mètres
    elevationGain: float   # en mètres
    splitTime: float       # en secondes
    pace: float            # en sec/km
    cumulativeTime: float  # en secondes
    calories: float
    carbsNeeded: float
    hydrationNeeded: float # en mL

class RacePlan(BaseModel):
    """ Modèle principal pour la réponse de l'API. """
    plan: List[Segment]
    totalTime: float         # en secondes
    totalDistance: float     # en km
    totalElevation: float    # en mètres
    totalCalories: float
    hourlyHydration: float   # en mL/h

