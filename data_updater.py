import pandas as pd
import requests
import os
from datetime import datetime, timedelta
import numpy as np

# Coordinates for cities
CITY_COORDS = {
    "Karachi": {"lat": 24.8607, "lon": 67.0011},
    "Lahore": {"lat": 31.5204, "lon": 74.3587},
    "Islamabad": {"lat": 33.6844, "lon": 73.0479},
    "Quetta": {"lat": 30.1798, "lon": 66.9750}
}

def fetch_open_meteo_data(city, lat, lon, start_date, end_date, static_attrs=None):
    """
    Fetches hourly data from Open-Meteo and aggregates it to daily to match the schema.
    """
    if static_attrs is None:
        static_attrs = {}
        
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    # We need specific hourly variables to aggregate into our daily columns
    # Schema mapping:
    # tmax -> daily max of temperature_2m
    # tmin -> daily min of temperature_2m
    # tavg -> daily mean of temperature_2m
    # prcp -> daily sum of precipitation
    # wspd -> daily mean of wind_speed_10m
    # humidity -> daily mean of relative_humidity_2m
    # pressure -> daily mean of surface_pressure
    # dew_point -> daily mean of dewpoint_2m
    # cloud_cover -> daily mean of cloud_cover
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,pressure_msl,cloud_cover,wind_speed_10m",
        "timezone": "auto"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        hourly = data.get("hourly", {})
        df_hourly = pd.DataFrame(hourly)
        df_hourly["time"] = pd.to_datetime(df_hourly["time"])
        df_hourly["date"] = df_hourly["time"].dt.normalize() # Keep as datetime64
        
        # Aggregation
        daily_stats = df_hourly.groupby("date").agg({
            "temperature_2m": ["max", "min", "mean"],
            "precipitation": "sum",
            "wind_speed_10m": "mean",
            "relative_humidity_2m": "mean",
            "pressure_msl": "mean",
            "dew_point_2m": "mean",
            "cloud_cover": "mean"
        })
        
        # Flatten columns
        daily_stats.columns = ['_'.join(col).strip() for col in daily_stats.columns.values]
        daily_stats = daily_stats.reset_index()
        
        # Rename to match our CSV schema
        # Schema: date,tmax,tmin,tavg,prcp,wspd,humidity,pressure,dew_point,cloud_cover,city,region,latitude,longitude,elevation
        
        df_final = pd.DataFrame()
        df_final["date"] = daily_stats["date"]
        df_final["tmax"] = daily_stats["temperature_2m_max"]
        df_final["tmin"] = daily_stats["temperature_2m_min"]
        df_final["tavg"] = daily_stats["temperature_2m_mean"]
        df_final["prcp"] = daily_stats["precipitation_sum"]
        # Scaling Wind Speed by 2.1 to match historical distribution (Historical Mean ~16.6 vs Open-Meteo ~7.9)
        df_final["wspd"] = daily_stats["wind_speed_10m_mean"] * 2.1
        df_final["humidity"] = daily_stats["relative_humidity_2m_mean"]
        df_final["pressure"] = daily_stats["pressure_msl_mean"]
        df_final["dew_point"] = daily_stats["dew_point_2m_mean"]
        df_final["cloud_cover"] = daily_stats["cloud_cover_mean"]
        
        # Add static columns
        df_final["city"] = city
        df_final["latitude"] = static_attrs.get("latitude", lat)
        df_final["longitude"] = static_attrs.get("longitude", lon)
        df_final["region"] = static_attrs.get("region", "Unknown")
        df_final["elevation"] = static_attrs.get("elevation", 0.0)
        
        return df_final
        
    except Exception as e:
        print(f"Error fetching data for {city}: {e}")
        return pd.DataFrame()

def update_weather_dataset(csv_path):
    """
    Main function to update the dataset.
    """
    if not os.path.exists(csv_path):
        return "CSV file not found."
        
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df['city'] = df['city'].astype(str).str.strip().str.title()
    
    new_rows = []
    updated_cities = []
    
    today = datetime.now().date()
    # Open-Meteo archive usually has data up to yesterday or today. 
    # We'll try to fetch up to yesterday to be safe, or today if available.
    target_end_date = today - timedelta(days=1) 
    
    for city, coords in CITY_COORDS.items():
        city_df = df[df['city'] == city]
        if city_df.empty:
            continue
            
        last_date = city_df['date'].max().date()
        
        if last_date < target_end_date:
            start_date = last_date + timedelta(days=1)
            print(f"Updating {city}: {start_date} to {target_end_date}")
            
            # Get static attributes from the last row of existing data
            last_row = city_df.iloc[-1]
            static_attrs = {
                "region": last_row.get("region", "Unknown"),
                "elevation": last_row.get("elevation", 0.0),
                "latitude": last_row.get("latitude", coords["lat"]),
                "longitude": last_row.get("longitude", coords["lon"])
            }
            
            fetched_df = fetch_open_meteo_data(
                city, 
                coords["lat"], 
                coords["lon"], 
                start_date.strftime("%Y-%m-%d"), 
                target_end_date.strftime("%Y-%m-%d"),
                static_attrs
            )
            
            if not fetched_df.empty:
                new_rows.append(fetched_df)
                updated_cities.append(city)
    
    if new_rows:
        new_data = pd.concat(new_rows, ignore_index=True)
        # Ensure column order matches
        # Get cols from original df
        cols = df.columns.tolist()
        # Align new_data to these cols
        for c in cols:
            if c not in new_data.columns:
                new_data[c] = 0 # or appropriate default
        
        new_data = new_data[cols]
        
        # Append and Save
        df_updated = pd.concat([df, new_data], ignore_index=True)
        df_updated = df_updated.sort_values(['city', 'date']).reset_index(drop=True)
        df_updated.to_csv(csv_path, index=False)
        
        # Delete preprocessed cache to force regeneration
        cache_file = "data_master_fixed_preprocessed.csv"
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
                print(f"Deleted stale cache: {cache_file}")
            except Exception as e:
                print(f"Warning: Could not delete cache: {e}")
        
        return f"Successfully updated data for: {', '.join(updated_cities)}"
    else:
        return "Data is already up to date."
