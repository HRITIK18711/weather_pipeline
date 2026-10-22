
import os
import time
import random
import uuid

import requests
import pandas as pd

from google.cloud import bigquery
from google.api_core.exceptions import (
    GoogleAPICallError,
    GoogleAPIError,
    ServiceUnavailable,
    DeadlineExceeded,
    InternalServerError,
    BadGateway,
    TooManyRequests,
    NotFound,
)

# ==========================================
# 1. CONFIGURATION
# ==========================================
PROJECT_ID = "modified-link-512401-i9"
DATASET_ID = "global_weather_data"
TABLE_ID = "current_weather_stats"
LOCATION = "US"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "data", "population.csv")
)

KEY_PATH = os.path.join(BASE_DIR, "gcp-key.json")

TABLE_REF = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

MAX_UPLOAD_ATTEMPTS = 5


# ==========================================
# 2. LOAD POPULATION DATA
# ==========================================
print("\n[1/5] Loading population data...")

if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(
        f"Population CSV not found: {DATA_PATH}"
    )

df_pop = pd.read_csv(DATA_PATH)

required_cols = {
    "City", "Country", "Latitude", "Longitude"
}

missing_cols = required_cols - set(df_pop.columns)

if missing_cols:
    raise ValueError(
        f"Missing required CSV columns: {sorted(missing_cols)}"
    )

if df_pop.empty:
    raise ValueError("population.csv contains no data.")

df_targets = df_pop.head(50).copy()

# Preserve a unique row identifier so cities with the same
# name are not accidentally merged with one another.
df_targets = df_targets.reset_index(drop=True)
df_targets["_row_id"] = range(len(df_targets))

df_targets["City"] = (
    df_targets["City"].astype("string").str.strip()
)

df_targets["Country"] = (
    df_targets["Country"].astype("string").str.strip()
)

for column in ["Latitude", "Longitude"]:
    df_targets[column] = pd.to_numeric(
        df_targets[column], errors="coerce"
    )

df_targets = df_targets.dropna(
    subset=["City", "Latitude", "Longitude"]
).copy()

df_targets = df_targets[
    df_targets["City"].ne("")
    & df_targets["Latitude"].between(-90, 90)
    & df_targets["Longitude"].between(-180, 180)
].copy()

if df_targets.empty:
    raise ValueError("No cities have valid coordinates.")

print(f"Cities selected: {len(df_targets)}")


# ==========================================
# 3. FETCH LIVE WEATHER DATA
# ==========================================
print("\n[2/5] Fetching live weather data...")

weather_records = []

retryable_http_codes = {429, 500, 502, 503, 504}

with requests.Session() as session:

    for _, row in df_targets.iterrows():

        city = row["City"]
        row_id = row["_row_id"]

        params = {
            "latitude": float(row["Latitude"]),
            "longitude": float(row["Longitude"]),
            "current_weather": "true"
        }

        for api_attempt in range(1, 4):

            try:
                response = session.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params=params,
                    timeout=25
                )

                if response.status_code in retryable_http_codes:
                    response.raise_for_status()

                response.raise_for_status()

                data = response.json()
                current = data.get("current_weather")

                if not current:
                    print(f"WARNING: No current weather for {city}")
                    break

                temperature = current.get("temperature")

                if temperature is None:
                    print(f"WARNING: Missing temperature for {city}")
                    break

                weather_records.append({
                    "_row_id": row_id,
                    "Temperature": temperature,
                    "WindSpeed": current.get("windspeed"),
                    "WeatherCode": current.get("weathercode")
                })

                break

            except requests.HTTPError as e:
                status = (
                    e.response.status_code
                    if e.response is not None else None
                )

                if (
                    status in retryable_http_codes
                    and api_attempt < 3
                ):
                    delay = 2 ** api_attempt
                    print(
                        f"Temporary API error for {city}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    continue

                print(f"WARNING: API request failed for {city}: {e}")
                break

            except (requests.RequestException, ValueError) as e:
                if api_attempt < 3:
                    time.sleep(2 ** api_attempt)
                    continue

                print(f"WARNING: Could not fetch {city}: {e}")
                break

        time.sleep(0.1)

if not weather_records:
    raise RuntimeError(
        "No weather data was retrieved. Check your internet "
        "connection and the Open-Meteo service."
    )

df_weather = pd.DataFrame(weather_records)

print(f"Weather records retrieved: {len(df_weather)}")


# ==========================================
# 4. MERGE, CLEAN AND TRANSFORM DATA
# ==========================================
print("\n[3/5] Cleaning and processing data...")

df_merged = pd.merge(
    df_targets,
    df_weather,
    on="_row_id",
    how="inner",
    validate="one_to_one"
)

df_cleaned = df_merged.drop(
    columns=["_row_id"]
).copy()

for column in [
    "Latitude", "Longitude", "Temperature", "WindSpeed"
]:
    df_cleaned[column] = pd.to_numeric(
        df_cleaned[column], errors="coerce"
    )

df_cleaned = df_cleaned.dropna(
    subset=["Temperature", "Latitude", "Longitude"]
).copy()

if df_cleaned.empty:
    raise ValueError("No valid records remain after cleaning.")

# Population: handle absent, invalid and missing values.
if "Population" not in df_cleaned.columns:
    print("WARNING: Population column missing; using fallback.")
    df_cleaned["Population"] = 15000000

population = (
    df_cleaned["Population"]
    .astype("string")
    .str.replace(",", "", regex=False)
    .str.strip()
)

df_cleaned["Population"] = pd.to_numeric(
    population, errors="coerce"
).fillna(0)

df_cleaned["Population"] = (
    df_cleaned["Population"]
    .clip(lower=0)
    .astype("int64")
)

# Keep nullable weather fields compatible with BigQuery.
df_cleaned["WindSpeed"] = df_cleaned["WindSpeed"].astype("float64")

df_cleaned["WeatherCode"] = pd.to_numeric(
    df_cleaned["WeatherCode"], errors="coerce"
).astype("Int64")

df_cleaned["Temperature"] = df_cleaned["Temperature"].astype("float64")
df_cleaned["Latitude"] = df_cleaned["Latitude"].astype("float64")
df_cleaned["Longitude"] = df_cleaned["Longitude"].astype("float64")

df_cleaned["Temperature_Fahrenheit"] = (
    df_cleaned["Temperature"] * 9 / 5
) + 32


def categorize_temp(celsius):
    if celsius < 15:
        return "Cold"
    elif celsius <= 28:
        return "Mild"
    return "Hot"


df_cleaned["Temp_Category"] = (
    df_cleaned["Temperature"].apply(categorize_temp)
)

# Convert text columns to standard Python strings or None.
# This avoids nullable pandas string values complicating uploads.
for column in ["City", "Country"]:
    df_cleaned[column] = df_cleaned[column].map(
        lambda value: str(value) if pd.notna(value) else None
    ).astype(object)

# Replace missing numeric measurements with None where appropriate.
# BigQuery will store these as NULL.
df_cleaned["WindSpeed"] = df_cleaned["WindSpeed"].where(
    df_cleaned["WindSpeed"].notna(), None
)

df_cleaned["WeatherCode"] = df_cleaned["WeatherCode"].where(
    df_cleaned["WeatherCode"].notna(), None
)

df_cleaned = df_cleaned.reset_index(drop=True)

print("\nCleaned DataFrame preview:")
print(
    df_cleaned[
        [
            "City",
            "Country",
            "Population",
            "Temperature",
            "Temp_Category"
        ]
    ].head().to_string(index=False)
)

print(f"Total cleaned rows: {len(df_cleaned)}")


# ==========================================
# 5. AUTHENTICATE WITH BIGQUERY
# ==========================================
print("\n[4/5] Authenticating with BigQuery...")

if not os.path.isfile(KEY_PATH):
    raise FileNotFoundError(
        f"Credentials file not found: {KEY_PATH}"
    )

try:
    client = bigquery.Client.from_service_account_json(
        KEY_PATH,
        project=PROJECT_ID,
        location=LOCATION
    )

    print("BigQuery client initialized successfully.")

except Exception as e:
    raise RuntimeError(
        f"Could not initialize BigQuery client: {e}"
    ) from e


# ==========================================
# 6. VERIFY DATASET ACCESS
# ==========================================
print("\nChecking BigQuery dataset...")

try:
    dataset = client.get_dataset(
        f"{PROJECT_ID}.{DATASET_ID}"
    )

    if dataset.location.upper() != LOCATION.upper():
        raise ValueError(
            f"Dataset location is {dataset.location}; "
            f"configured location is {LOCATION}."
        )

    print(f"Dataset verified: {DATASET_ID}")
    print("Dataset access confirmed.")

except NotFound as e:
    raise RuntimeError(
        f"Dataset {DATASET_ID} was not found. Check the "
        "project, dataset name and service account permissions."
    ) from e

except GoogleAPICallError as e:
    raise RuntimeError(
        f"Could not access BigQuery dataset: {e}"
    ) from e


# ==========================================
# 7. UPLOAD DATA WITH RETRIES
# ==========================================
print("\n[5/5] Uploading data to BigQuery...")
print(f"Destination: {TABLE_REF}")

job_config = bigquery.LoadJobConfig(
    write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
)

retryable_errors = (
    ServiceUnavailable,
    DeadlineExceeded,
    InternalServerError,
    BadGateway,
    TooManyRequests
)

upload_success = False

for attempt in range(1, MAX_UPLOAD_ATTEMPTS + 1):

    # Use a unique job ID for each attempt. This helps avoid
    # accidental job-ID collisions after a failed request.
    job_id = f"weather_load_{uuid.uuid4().hex}"

    try:
        print(
            f"Upload attempt {attempt}/{MAX_UPLOAD_ATTEMPTS}..."
        )

        job = client.load_table_from_dataframe(
            df_cleaned,
            TABLE_REF,
            job_config=job_config,
            job_id=job_id,
            timeout=120,
            num_retries=3
        )

        job.result(timeout=300)

        upload_success = True

        print("\nSUCCESS: Data uploaded to BigQuery.")
        print(f"Table: {TABLE_REF}")
        print(f"Rows uploaded: {job.output_rows}")
        print(f"Job ID: {job.job_id}")

        break

    except retryable_errors as e:

        if attempt == MAX_UPLOAD_ATTEMPTS:
            print(
                f"ERROR: Upload failed after "
                f"{MAX_UPLOAD_ATTEMPTS} attempts."
            )
            raise

        delay = min(2 ** attempt, 30) + random.uniform(0, 1)

        print(f"Temporary BigQuery service/network error: {e}")
        print(f"Retrying in {delay:.1f} seconds...")

        time.sleep(delay)

    except GoogleAPICallError as e:
        print(f"BigQuery API error: {e}")
        raise

    except Exception as e:
        print(f"Unexpected upload error: {e}")
        raise

if not upload_success:
    raise RuntimeError("BigQuery upload did not complete.")

print("\nWeather Pipeline completed successfully!")