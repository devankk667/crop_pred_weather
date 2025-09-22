from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import requests
from geopy.geocoders import Nominatim

from schemas import PredictionRequest, PredictionResponse
from model import predict_yield, MODEL_VERSION

app = FastAPI(title="Crop Yield Prediction API", version=MODEL_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------
# Frontend mounting (frontend folder is in project root)
# ------------------
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(index_file.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="Frontend not found")

# ------------------
# Helper functions
# ------------------
def geocode_location(state: str, district: str):
    geolocator = Nominatim(user_agent="crop_pred_weather")
    location = geolocator.geocode(f"{district}, {state}, India")
    if not location:
        raise HTTPException(status_code=400, detail="Location not found")
    return location.latitude, location.longitude

def fetch_weather(lat: float, lon: float, year: int, season: str):
    season_months = {"kharif": [6, 10], "rabi": [11, 3], "zaid": [4, 5]}
    if season.lower() not in season_months:
        raise HTTPException(status_code=400, detail="Invalid season")

    start_month, end_month = season_months[season.lower()]
    if season.lower() == "rabi":
        start_date = f"{year-1}-11-01"
        end_date = f"{year}-03-31"
    else:
        start_date = f"{year}-{start_month:02d}-01"
        end_date = f"{year}-{end_month:02d}-28"

    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily=temperature_2m_max,temperature_2m_min,"
        f"precipitation_sum,windspeed_10m_max,"
        f"relative_humidity_2m_max,relative_humidity_2m_min"
        f"&timezone=auto"
    )

    r = requests.get(url)
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail="Weather API failed")
    
    data = r.json()
    if "daily" not in data:
        raise HTTPException(status_code=500, detail="Weather data not available")
    
    daily = data["daily"]

    # summary averages
    avg_temp = sum([(tmax+tmin)/2 for tmax, tmin in zip(daily["temperature_2m_max"], daily["temperature_2m_min"])]) / len(daily["temperature_2m_max"])
    total_precip = sum(daily["precipitation_sum"])
    avg_humidity = sum([(hmax+hmin)/2 for hmax, hmin in zip(daily["relative_humidity_2m_max"], daily["relative_humidity_2m_min"])]) / len(daily["relative_humidity_2m_max"])
    avg_windspeed = sum(daily["windspeed_10m_max"]) / len(daily["windspeed_10m_max"])

    return {
        "daily": daily,  # full weather arrays
        "summary": {
            "avg_temp": avg_temp,
            "total_precip": total_precip,
            "avg_humidity": avg_humidity,
            "avg_windspeed": avg_windspeed
        }
    }

# ------------------
# Endpoints
# ------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    try:
        lat, lon = geocode_location(req.state_name, req.district_name)
        weather = fetch_weather(lat, lon, req.year, req.season)
        input_dict = req.dict()
        input_dict.update(weather["summary"])  # feed only summary to model
        prediction = predict_yield(input_dict)
        return {
            "predicted_yield": round(prediction, 2),
            "model_version": MODEL_VERSION,
            "weather_used": weather
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
