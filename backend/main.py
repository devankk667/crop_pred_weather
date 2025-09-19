from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
# Helper functions
# ------------------

def geocode_location(state: str, district: str):
    geolocator = Nominatim(user_agent="crop_pred_weather")
    location = geolocator.geocode(f"{district}, {state}, India")
    if not location:
        raise HTTPException(status_code=400, detail="Location not found")
    return location.latitude, location.longitude

def fetch_weather(lat: float, lon: float, year: int, season: str):
    season_months = {
        "kharif": [6, 10],  # Jun–Oct
        "rabi": [11, 3],    # Nov–Mar
        "zaid": [4, 5],     # Apr–May
    }
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

    temps = [(tmax + tmin) / 2 for tmax, tmin in zip(
        data["daily"]["temperature_2m_max"],
        data["daily"]["temperature_2m_min"]
    )]
    avg_temp = sum(temps) / len(temps)

    total_precip = sum(data["daily"]["precipitation_sum"])

    humidities = [(hmax + hmin) / 2 for hmax, hmin in zip(
        data["daily"]["relative_humidity_2m_max"],
        data["daily"]["relative_humidity_2m_min"]
    )]
    avg_humidity = sum(humidities) / len(humidities)

    avg_windspeed = sum(data["daily"]["windspeed_10m_max"]) / len(data["daily"]["windspeed_10m_max"])

    return {
        "avg_temp": avg_temp,
        "total_precip": total_precip,
        "avg_humidity": avg_humidity,
        "avg_windspeed": avg_windspeed
    }

# ------------------
# Endpoints
# ------------------

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    try:
        # 1. Geocode
        lat, lon = geocode_location(req.state_name, req.district_name)

        # 2. Weather
        weather = fetch_weather(lat, lon, req.year, req.season)

        # 3. Prepare input for model
        input_dict = req.dict()
        input_dict.update(weather)

        # 4. Predict
        prediction = predict_yield(input_dict)

        return PredictionResponse(
            predicted_yield=round(prediction, 2),
            model_version=MODEL_VERSION,
            weather_used=weather
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
