from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class StravaActivity(BaseModel):
    id: int
    name: str
    distance: float
    moving_time: int
    total_elevation_gain: float
    start_date_local: str
    workout_type: Optional[int] = None
    suffer_score: Optional[float] = None

class AthleteProfile(BaseModel):
    vam: float
    downhill_technique_speed: float
    aerobic_endurance_speed: float
    fatigue_resistance: float
    pace_management: float
    max_hr: Optional[int] = None
    resting_hr: Optional[int] = None
    runner_type: str
    radar_data: Dict[str, float]
    analysis_level: str

class AidStation(BaseModel):
    distance: float
    duration: int
    name: Optional[str] = None

class PredictionInput(BaseModel):
    weight: float
    tech_multiplier: float
    temperature: int
    carb_intake: int # g/h
    manual_activity_ids: Optional[List[int]] = []
    aid_stations: Optional[List[Dict[str, Any]]] = []

class Segment(BaseModel):
    name: str
    distance: float
    elevation_gain: float
    time: float
    pace: float
    cumulative_time: float
    calories: float
    carbs_needed: float
    hydration_needed: float

class RacePlan(BaseModel):
    plan: List[Segment]
    total_time: float
    total_distance: float
    total_elevation: float
    total_calories: float
    total_carbs: float
    total_hydration: float

