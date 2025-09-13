import requests
import json

# Sample input data
sample_data = {
    "temperature": 25.5,
    "precipitation": 1200.0,
    "nitrogen": 150.0,
    "phosphorus": 30.0,
    "potassium": 200.0,
    "ph": 6.5,
    "crop_type": "Wheat",
    "planting_date": "2023-01-15",
    "harvest_date": "2023-06-15",
    "soil_type": "Loam"
}

# API endpoint
url = "http://localhost:8000/predict"

# Make the request
try:
    print("Sending prediction request...")
    response = requests.post(url, json=sample_data)
    
    # Check if the request was successful
    if response.status_code == 200:
        print("\nPrediction successful!")
        print("-" * 40)
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"\nError: {response.status_code}")
        print(response.text)

except requests.exceptions.RequestException as e:
    print(f"\nError making request: {e}")
    print("\nMake sure the FastAPI server is running with 'uvicorn app:app --reload'")

print("\nYou can also test the API using the interactive docs at: http://localhost:8000/docs")
