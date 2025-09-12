import os
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
from datetime import datetime, timedelta
import time
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Constants
BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMS = {
    'parameters': 'T2M,RH2M,PRECTOTCORR,WS2M',
    'community': 'AG',
    'format': 'JSON',
    'header': 'true',
    'time-standard': 'lst'
}

# Number of worker threads (adjust based on your system)
MAX_WORKERS = min(32, (os.cpu_count() or 1) * 4)

# Data directories
RAW_DATA_DIR = Path('weather_data/raw')
PROCESSED_DATA_DIR = Path('weather_data/processed')
SEASONAL_DATA_DIR = Path('weather_data/seasonal')

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SEASONAL_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Define Indian seasons
# Define Indian seasons with overlapping date ranges (month, day)
# Note: These ranges include overlapping periods common in Indian agriculture
SEASONS = {
    # Winter: Nov to Feb (overlaps with Rabi)
    'winter': [(11, 1), (2, 28)],  # Nov 1 - Feb 28
    # Summer: Mar to Jun (overlaps with Kharif)
    'summer': [(3, 1), (6, 15)],   # Mar 1 - Jun 15
    # Kharif: May to Oct (overlaps with Summer and Rabi)
    'kharif': [(4, 1), (10, 31)],  # May 1 - Oct 31
    # Rabi: Oct to Mar (overlaps with Winter and Kharif)
    'rabi': [(10, 1), (3, 31)]     # Oct 1 - Mar 31 (next year)
}

def is_date_in_season(date, season):
    """Check if a date falls within a season's date range."""
    month, day = date.month, date.day
    start_month, start_day = SEASONS[season][0]
    end_month, end_day = SEASONS[season][1]
    
    # Handle seasons that cross year boundaries (like rabi)
    if start_month > end_month:  # Season crosses year boundary
        if (month == start_month and day >= start_day) or \
           (month == end_month and day <= end_day) or \
           (month > start_month) or (month < end_month):
            return True
    else:  # Season is within the same year
        if (month == start_month and day >= start_day) or \
           (month == end_month and day <= end_day) or \
           (start_month < month < end_month):
            return True
    return False

def get_coordinates():
    """Load all district coordinates."""
    try:
        coords_df = pd.read_csv('district_coordinates.csv')
        # Convert to list of dicts for easier processing
        return coords_df.to_dict('records')
    except Exception as e:
        print(f"Error reading coordinates file: {e}")
        return []

def fetch_weather_data(district_data, year):
    """Fetch weather data for a single district and year."""
    district = district_data['district']
    state = district_data['state']
    lat = district_data['latitude']
    lon = district_data['longitude']
    
    # Check if we already have this data
    output_file = RAW_DATA_DIR / f"{district}_{state}_{year}.json"
    if output_file.exists():
        print(f"Skipping {district}, {state} - {year} (already exists)")
        return True
    
    # Format dates for the API
    start_date = f"{year}0101"
    end_date = f"{year}1231"
    
    params = PARAMS.copy()
    params.update({
        'start': start_date,
        'end': end_date,
        'latitude': lat,
        'longitude': lon
    })
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Save raw data
        with open(output_file, 'w') as f:
            json.dump(data, f)
            
        return True
    except Exception as e:
        print(f"Error fetching data for {district}, {state} - {year}: {e}")
        return False

def process_seasonal_data(file_path):
    """Process a single raw data file into seasonal data."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        if not data or 'properties' not in data or 'parameter' not in data['properties']:
            return None
            
        # Extract data
        params = data['properties']['parameter']
        dates = list(params[list(params.keys())[0]].keys())
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': pd.to_datetime(dates),
            'temperature': [params['T2M'].get(date, np.nan) for date in dates],
            'humidity': [params['RH2M'].get(date, np.nan) for date in dates],
            'precipitation': [params['PRECTOTCORR'].get(date, np.nan) for date in dates],
            'windspeed': [params['WS2M'].get(date, np.nan) for date in dates]
        })
        
        # Process each year in the data
        years = df['date'].dt.year.unique()
        all_seasonal_data = []
        
        for year in years:
            year_df = df[df['date'].dt.year == year].copy()
            
            # Process each season
            for season in SEASONS.keys():
                season_mask = year_df['date'].apply(lambda x: is_date_in_season(x, season))
                season_df = year_df[season_mask].copy()
                
                if len(season_df) == 0:
                    continue
                    
                # Calculate seasonal aggregates
                seasonal_data = {
                    'district': file_path.stem.split('_')[0],
                    'state': ' '.join(file_path.stem.split('_')[1:-1]),
                    'year': year,
                    'season': season,
                    'avg_temp': season_df['temperature'].mean(),
                    'total_precip': season_df['precipitation'].sum(),
                    'avg_humidity': season_df['humidity'].mean(),
                    'avg_windspeed': season_df['windspeed'].mean(),
                    'data_points': len(season_df)
                }
                all_seasonal_data.append(seasonal_data)
        
        return all_seasonal_data
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_raw_files():
    """Process all raw data files into seasonal data."""
    raw_files = list(RAW_DATA_DIR.glob('*.json'))
    all_seasonal_data = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Process files in parallel
        futures = [executor.submit(process_seasonal_data, file_path) for file_path in raw_files]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            result = future.result()
            if result:
                all_seasonal_data.extend(result)
    
    # Save all seasonal data to a single file
    if all_seasonal_data:
        df = pd.DataFrame(all_seasonal_data)
        output_file = SEASONAL_DATA_DIR / 'all_seasonal_data.csv'
        df.to_csv(output_file, index=False)
        print(f"\nSaved seasonal data to {output_file}")
        
        # Also save by year for easier access
        for year, group in df.groupby('year'):
            year_file = SEASONAL_DATA_DIR / f'seasonal_data_{year}.csv'
            group.to_csv(year_file, index=False)
    
    return all_seasonal_data

def main():
    # Get all district coordinates
    print("Loading district coordinates...")
    districts = get_coordinates()
    if not districts:
        print("No district coordinates found. Please check the coordinates file.")
        return
    
    print(f"Found {len(districts)} districts with coordinates")
    
    # Years to process (2015-2020 inclusive)
    years = list(range(2015, 2021))
    
    # Step 1: Download raw data for all districts and years
    print("\n=== Starting data download ===")
    all_tasks = []
    for district_data in districts:
        for year in years:
            all_tasks.append((district_data, year))
    
    # Process downloads in parallel with a progress bar
    print(f"\nDownloading data for {len(districts)} districts across {len(years)} years...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = [executor.submit(download_year, *task) for task in all_tasks]
        
        # Process results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Download progress"):
            try:
                future.result()
            except Exception as e:
                print(f"Error in download task: {e}")
    
    # Verify all files were downloaded
    raw_files = list(RAW_DATA_DIR.glob('*.json'))
    print(f"\nDownloaded {len(raw_files)} raw data files")
    
    # Step 2: Process raw data into seasonal data
    print("\n=== Processing data into seasonal aggregates ===")
    process_raw_files()
    
    print("\n=== Data processing complete ===")

def download_year(district_data, year):
    """Download weather data for a single district and year."""
    return fetch_weather_data(district_data, year)

if __name__ == "__main__":
    main()
