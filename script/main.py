import os
import requests
import pandas as pd
from google.cloud import bigquery

# ==========================================
# 1. Load Local Population Dataset
# ==========================================
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "population.csv")
print(f"Loading population data from: {DATA_PATH}")

df_pop = pd.read_csv(DATA_PATH)

# Verify required columns exist
required_cols = {"City", "Country", "Latitude", "Longitude"}
if not required_cols.issubset(df_pop.columns):
    raise ValueError(f"CSV must contain: {required_cols}")

# Limit to top 50 cities if dataset contains more
df_targets = df_pop.head(50).copy()

# ==========================================
# 2. Fetch Live Weather Data (Open-Meteo)
# ==========================================
weather_records = []
print("Fetching live weather for 50 cities...")

for _, row in df_targets.iterrows():
    lat = row["Latitude"]
    lon = row["Longitude"]
    city = row["City"]
    
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            current = resp.json().get("current_weather", {})
            weather_records.append({
                "City": city,
                "Temperature": current.get("temperature"),
                "WindSpeed": current.get("windspeed"),
                "WeatherCode": current.get("weathercode")
            })
    except Exception as e:
        print(f"Error fetching data for {city}: {e}")

df_weather = pd.DataFrame(weather_records)

# ==========================================
# 3. Merge Datasets
# ==========================================
df_merged = pd.merge(df_targets, df_weather, on="City", how="inner")

# ==========================================
# 4. Data Cleaning, Imputation & Feature Engineering
# ==========================================
# Drop rows where essential coordinates or temperature failed to fetch
df_cleaned = df_merged.dropna(subset=["Temperature", "Latitude", "Longitude"]).copy()

# Ensure population column exists and enforce int64
if "Population" not in df_cleaned.columns:
    df_cleaned["Population"] = 15000000  # Fallback default if population.csv omitted it
df_cleaned["Population"] = df_cleaned["Population"].fillna(0).astype("int64")

# Enforce float64 types
df_cleaned["Temperature"] = df_cleaned["Temperature"].astype("float64")
df_cleaned["WindSpeed"] = df_cleaned["WindSpeed"].astype("float64")
df_cleaned["Latitude"] = df_cleaned["Latitude"].astype("float64")
df_cleaned["Longitude"] = df_cleaned["Longitude"].astype("float64")

# Temperature conversion and classification
df_cleaned["Temperature_Fahrenheit"] = (df_cleaned["Temperature"] * 9/5) + 32

def categorize_temp(celsius):
    if celsius < 15.0:
        return "Cold"
    elif celsius <= 28.0:
        return "Mild"
    else:
        return "Hot"

df_cleaned["Temp_Category"] = df_cleaned["Temperature"].apply(categorize_temp)

print("\n--- Cleaned DataFrame Preview ---")
print(df_cleaned[["City", "Country", "Population", "Temperature", "Temp_Category"]].head())

# ==========================================
# 5. BigQuery Upload
# ==========================================
KEY_PATH = os.path.join(os.path.dirname(__file__), "gcp-key.json")
PROJECT_ID = "weather-demographics-bootcamp"
DATASET_ID = "global_weather_data"
TABLE_ID = "current_weather_stats"

if os.path.exists(KEY_PATH):
    print("\nAuthenticating with BigQuery...")
    client = bigquery.Client.from_service_account_json(KEY_PATH, project=PROJECT_ID)
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
    )

    job = client.load_table_from_dataframe(df_cleaned, table_ref, job_config=job_config)
    job.result()  # Wait for upload to complete
    print(f"Successfully loaded {len(df_cleaned)} rows into {table_ref}.")
else:
    print(f"\n[Notice] {KEY_PATH} not found. Local data processing passed successfully.")
