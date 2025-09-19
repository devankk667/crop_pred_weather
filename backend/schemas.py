from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    year: int = Field(..., ge=1900, le=2100)
    season: str
    crop: str
    state_name: str
    district_name: str

class PredictionResponse(BaseModel):
    predicted_yield: float
    model_version: str
    weather_used: dict
