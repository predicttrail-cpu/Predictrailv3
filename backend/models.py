from pydantic import BaseModel
from typing import List, Optional
    
    # --- Modèles pour les entrées de l'API ---
    
class ReferencePerformance(BaseModel):
        distance: float
        time: str
        elevGain: int
        elevLoss: int
        tech: float
    
class AidStation(BaseModel):
        name: str
        cumulativeDistance: float
    
class PredictionInput(BaseModel):
        weight: float
        gender: str
        references: List[ReferencePerformance]
        techMultiplier: float
        fatigueFactor: float
        carbIntake: int
        splitDistance: Optional[float] = None
        aidStations: Optional[List[AidStation]] = None
    
    # --- Modèles pour les sorties de l'API ---
    
class Segment(BaseModel):
        name: str
        distance: float
        elevationGain: float
        splitTime: float
        pace: float
        cumulativeTime: float
        calories: float
        carbsNeeded: float
    
class RacePlan(BaseModel):
        plan: List[Segment]
        totalTime: float
        totalDistance: float
        totalElevation: float
        totalCalories: float
    
    # --- NOUVEAUX MODÈLES POUR LES ACTIVITÉS STRAVA ---
    
class StravaActivity(BaseModel):
        """Définit les informations clés d'une activité Strava."""
        id: int
        name: str
        distance: float  # en mètres
        moving_time: int # en secondes
        total_elevation_gain: float # en mètres
        start_date_local: str
    

