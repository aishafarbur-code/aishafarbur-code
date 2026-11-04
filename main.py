import io
import csv
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from database import init_db, get_all_steps_status, update_step_status, get_db
from weather_service import fetch_current_weather_api, fetch_historical_and_forecast_data, get_raw_weather_records
from cleaner import clean_and_prepare_dataset, get_latest_cleaning_status
from ml_engine import train_and_evaluate_models, predict_single, get_active_model_details, MODEL_EXPLANATIONS
from monitor import get_monitoring_dashboard_data, simulate_real_actual_comparison

app = FastAPI(
    title="Smart Weather Station & Machine Learning System API",
    description="Backend API loogu talagalay Barashada Machine Learning iyo Saadaasha Cimilada Muqdisho",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    init_db()

# Request Models
class CollectDataRequest(BaseModel):
    past_days: int = Field(30, ge=7, le=90, description="Muddada taariikhiga ah ee la soo ururinayo (7 ilaa 90 maalmood)")

class PredictRequest(BaseModel):
    temperature: float = Field(..., description="Heerkulka hadda (°C)")
    relative_humidity: float = Field(..., description="Qoyaanka hawada (%)")
    surface_pressure: float = Field(..., description="Cadaadiska hawada (hPa)")
    wind_speed: float = Field(..., description="Xawaaraha dabaysha (km/h)")

class VerifyPredictionRequest(BaseModel):
    prediction_id: int
    actual_temperature: float

# ==================== STEP 0: WORKFLOW STATUS ====================

@app.get("/api/status")
def get_workflow_status():
    """Soo celi xaaladda 7-da tallaabo ee Machine Learning Process"""
    return {
        "success": True,
        "steps": get_all_steps_status()
    }

# ==================== STEP 1: COLLECT DATA ====================

@app.get("/api/weather/current")
async def get_current_weather():
    """Ka soo qaad xogta cimilada tooska ah ee Muqdisho Open-Meteo API"""
    res = await fetch_current_weather_api()
    return res

@app.post("/api/weather/collect")
async def collect_weather_data(payload: CollectDataRequest):
    """Ururi xogta cimilada Muqdisho ee Open-Meteo API (Taariikh iyo Saadaal)"""
    res = await fetch_historical_and_forecast_data(past_days=payload.past_days)
    return res

@app.get("/api/weather/raw")
def list_raw_weather(limit: int = 150, offset: int = 0):
    """Liiska xogta ceeriin ee lagu kaydiyey SQLite database"""
    return get_raw_weather_records(limit=limit, offset=offset)

@app.get("/api/weather/export-csv")
def export_raw_weather_csv():
    """Soo dejiso xogta ceeriin qaab CSV ah (Download CSV button)"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT recorded_at, temperature, relative_humidity, surface_pressure, wind_speed, next_hour_temp, data_type, collected_at
        FROM raw_weather
        ORDER BY recorded_at ASC;
    """)
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        raise HTTPException(status_code=404, detail="Ma jirto xog la dhoofin karo. Marka hore ururi xogta.")
        
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["recorded_at", "temperature_C", "relative_humidity_pct", "surface_pressure_hPa", "wind_speed_kmh", "next_hour_temp_target_C", "data_type", "collected_at"])
    
    for r in rows:
        writer.writerow([
            r["recorded_at"],
            r["temperature"],
            r["relative_humidity"],
            r["surface_pressure"],
            r["wind_speed"],
            r["next_hour_temp"],
            r["data_type"],
            r["collected_at"]
        ])
        
    csv_content = output.getvalue()
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=muqdisho_weather_data.csv"}
    )

# ==================== STEP 2: PREPARE & CLEAN DATA ====================

@app.post("/api/clean/run")
def run_cleaning():
    """Nadiifi oo diyaari xogta cimilada (Missing values, Outliers, Chronological order)"""
    return clean_and_prepare_dataset()

@app.get("/api/clean/status")
def get_cleaning_info():
    """Xaaladda iyo natiijooyinkii ugu dambeeyay ee nadiifinta xogta"""
    return get_latest_cleaning_status()

# ==================== STEP 3: CHOOSE A MODEL ====================

@app.get("/api/models/info")
def get_models_info():
    """Sharaxaadda af-Soomaaliga ah ee 3-da model: Baseline, Linear Regression, Random Forest"""
    return {
        "success": True,
        "models": MODEL_EXPLANATIONS,
        "problem_type": "Regression (Saadaalinta qiime tiro ah: Heerkulka saacadda xigta)",
        "selection_criteria": "Model-ka waxaa lagu dooranayaa natiijada Validation Set (MAE & RMSE-ga ugu yar, R²-ka ugu sarreeya)."
    }

# ==================== STEP 4 & 5: TRAIN & EVALUATE ====================

@app.post("/api/models/train")
def train_models():
    """Tababar 3-da moodal adoo adeegsanaya scikit-learn iyo time-series split"""
    return train_and_evaluate_models()

@app.get("/api/models/active")
def get_active_model():
    """Soo celi model-ka hadda firfircoon iyo natiijooyinkii qiimaynta"""
    return get_active_model_details()

# ==================== STEP 6: DEPLOY THE MODEL ====================

@app.post("/api/predict")
def predict_weather(payload: PredictRequest):
    """Saadaali heerkulka saacadda xigta adoo adeegsanaya model-ka la tababaray"""
    return predict_single(
        temperature=payload.temperature,
        relative_humidity=payload.relative_humidity,
        surface_pressure=payload.surface_pressure,
        wind_speed=payload.wind_speed
    )

# ==================== STEP 7: MONITOR & IMPROVE ====================

@app.get("/api/monitor/dashboard")
def get_monitoring_data():
    """Soo celi xogta la socodka tayada model-ka, taariikhda iyo drift alerts"""
    return get_monitoring_dashboard_data()

@app.post("/api/monitor/verify")
def verify_prediction(payload: VerifyPredictionRequest):
    """Geli cabbirka dhabta ah si loo xisaabiyo qaladka dhabta ah ee saadaasha (Drift testing)"""
    return simulate_real_actual_comparison(
        prediction_id=payload.prediction_id,
        actual_value=payload.actual_temperature
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
