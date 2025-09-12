#!/usr/bin/env python3
"""
nasa_power_weather.py - Windows Compatible Version
Usage:
  python scripts\\nasa_power_weather.py --coords input\\district_coords.csv --start-year 2015 --end-year 2023 --out data\\raw\\weather_daily_combined.csv

district_coords.csv should be:
district,lat,lon
Varanasi,25.3176,82.9739
Lucknow,26.8467,80.9462
"""
import argparse, requests, pandas as pd, time, io, os
from tqdm import tqdm

POWER_BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"

# parameters to request
PARAMS = "T2M,T2M_MIN,T2M_MAX,PRECTOT,RH2M"  # mean temp, min, max, total precipitation, relative humidity

def fetch_power_for_range(lat, lon, start, end, district_name, year):
    url = (f"{POWER_BASE}?parameters={PARAMS}&community=AG&longitude={lon}&latitude={lat}"
           f"&start={start}&end={end}&format=CSV")
    
    print(f"  Fetching {district_name} {year}: {start}-{end}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    text = r.text
    
    # POWER returns comment lines starting with '#'; read with pandas ignoring comment lines
    try:
        df = pd.read_csv(io.StringIO(text), comment='#')
    except Exception:
        # fallback: try to find header line then read
        lines = text.splitlines()
        header_idx = 0
        for i, line in enumerate(lines):
            if "T2M" in line or "PRECTOT" in line:
                header_idx = i
                break
        df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    
    # identify date column
    date_col = None
    for c in df.columns:
        if 'DATE' in c.upper() or 'YYYY' in c.upper() or c.lower() == 'date':
            date_col = c; break
    if date_col is None:
        date_col = df.columns[0]
    
    df = df.rename(columns={date_col: "date"})
    
    # parse date formats
    if df['date'].dtype in [int, float] or df['date'].astype(str).str.match(r'^\d{8}$').all():
        df['date'] = pd.to_datetime(df['date'].astype(int).astype(str), format='%Y%m%d')
    else:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # map columns
    mapping = {}
    for c in df.columns:
        cu = c.upper()
        if "T2M_MIN" in cu or ("MIN" in cu and "T2M" in cu):
            mapping[c] = "temp_min"
        elif "T2M_MAX" in cu or ("MAX" in cu and "T2M" in cu):
            mapping[c] = "temp_max"
        elif cu == "T2M" or cu.endswith("T2M"):
            mapping[c] = "temp_mean"
        elif "PRECTOT" in cu or "PRECIP" in cu:
            mapping[c] = "precipitation_mm"
        elif "RH2M" in cu or "HUM" in cu:
            mapping[c] = "humidity"
    
    df = df.rename(columns=mapping)
    
    # keep only desired cols
    cols_keep = ["date","temp_min","temp_max","temp_mean","precipitation_mm","humidity"]
    for c in cols_keep:
        if c not in df.columns:
            df[c] = pd.NA
    
    return df[cols_keep]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coords", required=True, help="CSV mapping district,lat,lon")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--out", default="weather_daily_combined.csv")
    parser.add_argument("--season-months", default="6,7,8,9,10,11", help="comma list of months for season (default Kharif)")
    args = parser.parse_args()

    if not os.path.exists(args.coords):
        print(f"Error: Coordinates file not found: {args.coords}")
        print("Please create the file with format:")
        print("district,lat,lon")
        print("Varanasi,25.3176,82.9739")
        return

    coords_df = pd.read_csv(args.coords)
    print(f"Found {len(coords_df)} districts to fetch weather data for")
    season_months = [int(m) for m in args.season_months.split(",")]

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)

    all_records = []
    total_requests = len(coords_df) * (args.end_year - args.start_year + 1)
    
    with tqdm(total=total_requests, desc="Fetching weather data") as pbar:
        for _, row in coords_df.iterrows():
            district = row['district']
            lat = float(row['lat']); lon = float(row['lon'])
            
            print(f"\nProcessing {district} ({lat}, {lon})")
            
            # loop year by year to avoid massive request
            for year in range(args.start_year, args.end_year + 1):
                start = f"{year}0601"
                end = f"{year}1130"
                try:
                    df = fetch_power_for_range(lat, lon, start, end, district, year)
                    df['district'] = district
                    all_records.append(df)
                    time.sleep(0.5)  # be polite to NASA servers
                    pbar.update(1)
                except Exception as e:
                    print(f"  ✗ Failed for {district} {year}: {e}")
                    pbar.update(1)
                    time.sleep(1)

    if not all_records:
        print("No weather data fetched. Check your internet connection and coordinates.")
        return
    
    print(f"\nCombining {len(all_records)} datasets...")
    combined = pd.concat(all_records, ignore_index=True)
    combined.to_csv(args.out, index=False)
    print(f"✓ Saved combined daily weather to {args.out}")
    print(f"✓ Total records: {len(combined)}")
    
    # also save aggregated season sums/means by district-year
    print("Creating seasonal aggregates...")
    combined['year'] = combined['date'].dt.year
    combined['month'] = combined['date'].dt.month
    kharif = combined[combined['month'].isin(season_months)]
    
    agg = kharif.groupby(['district','year']).agg(
        temp_min_mean=('temp_min','mean'),
        temp_max_mean=('temp_max','mean'),
        temp_mean_mean=('temp_mean','mean'),
        precipitation_sum=('precipitation_mm','sum'),
        humidity_mean=('humidity','mean'),
        days_with_data=('date','count')
    ).reset_index()
    
    seasonal_out = args.out.replace('.csv', '_seasonal_aggregates.csv')
    agg.to_csv(seasonal_out, index=False)
    print(f"✓ Saved seasonal aggregates to {seasonal_out}")
    
    print(f"\nSample daily data:")
    print(combined.head())

if __name__ == "__main__":
    main()