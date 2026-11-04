import sqlite3
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
from database import get_db, update_step_status

DRIFT_THRESHOLD_MAE = 2.5  # Haddii qaladka celcelisku ka bato 2.5°C, digniin ayaa soo baxaysa

# models_registry column names
REG_COLS = ["id", "model_name", "algorithm", "hyperparameters", "train_samples", "val_samples", "test_samples",
            "val_mae", "val_rmse", "val_r2", "test_mae", "test_rmse", "test_r2", "model_file_path", "is_active", "trained_at"]

# predictions_log column names
PRED_COLS = ["id", "model_id", "model_algorithm", "input_temperature", "input_humidity",
             "input_wind_speed", "input_surface_pressure", "predicted_next_temp",
             "prediction_time", "actual_next_temp", "absolute_error", "status"]


def get_monitoring_dashboard_data() -> Dict[str, Any]:
    """
    Tallaabada 7: Monitor & Improve (La Socodka iyo Horumarinta)
    """
    conn = get_db()
    cursor = conn.cursor()
    
    # 1. Active model
    cursor.execute("SELECT * FROM models_registry WHERE is_active = 1 ORDER BY id DESC LIMIT 1;")
    active_row = cursor.fetchone()
    active = dict(zip(REG_COLS, active_row)) if active_row else None
    
    # 2. All training history
    cursor.execute("SELECT * FROM models_registry ORDER BY id DESC LIMIT 10;")
    training_history = [dict(zip(REG_COLS, r)) for r in cursor.fetchall()]
    
    # 3. Recent predictions log
    cursor.execute("""
        SELECT id, model_id, model_algorithm, input_temperature, input_humidity, input_wind_speed,
               input_surface_pressure, predicted_next_temp, prediction_time, actual_next_temp,
               absolute_error, status
        FROM predictions_log
        ORDER BY id DESC
        LIMIT 50;
    """)
    pred_rows = cursor.fetchall()
    
    pred_col_names = ["id", "model_id", "model_algorithm", "input_temperature", "input_humidity",
                      "input_wind_speed", "input_surface_pressure", "predicted_next_temp",
                      "prediction_time", "actual_next_temp", "absolute_error", "status"]
    predictions = [dict(zip(pred_col_names, r)) for r in pred_rows]
    
    # 4. Calculate error metrics on predictions that have actual values
    evaluated_predictions = [p for p in predictions if p["actual_next_temp"] is not None]
    
    current_live_mae = None
    drift_alert = False
    alert_message = None
    
    if evaluated_predictions:
        errors = [abs(float(p["actual_next_temp"]) - float(p["predicted_next_temp"])) for p in evaluated_predictions]
        current_live_mae = round(float(np.mean(errors)), 2)
        
        if current_live_mae > DRIFT_THRESHOLD_MAE:
            drift_alert = True
            alert_message = (
                f"Digniin Drift: Celceliska qaladka saadaasha tooska ah ({current_live_mae}°C) "
                f"wuxuu dhaafay heerka la oggol yahay ({DRIFT_THRESHOLD_MAE}°C). "
                f"Waxaa lagu talinayaa in model-ka dib loo tababaro iyadoo la adeegsanayo xog cusub."
            )
            update_step_status(7, "in_progress", f"Digniin Drift: Qaladka hadda waa {current_live_mae}°C.")
        else:
            update_step_status(7, "completed", f"Nidaamka kormeerku wuxuu xaqiijinayaa in model-ku si fiican u shaqaynayo (MAE={current_live_mae}°C).")
    else:
        update_step_status(7, "completed", "Kormeerka tooska ah waa firfircoon yahay; wuxuu diyaar u yahay isbarbardhigga.")
    
    conn.close()
    
    educational_reasons = [
        {
            "ciwaan": "Isbeddelka Xilliyada (Seasonal Climate Shift / Concept Drift)",
            "sharaxaad": "Cimilada Muqdisho waxay leedahay xilliyaal kala duwan sida Xagaa, Dayr, Jilaal iyo Gu. Model lagu tababaray xilliga Xagaaga ma saadaalin karo xilliga roobabka Gu'ga si sax ah sababtoo ah qaabka qoyaanka iyo roobka ayaa isbeddelay."
        },
        {
            "ciwaan": "Xog Cusub oo Soo Kordhaysa (New Incoming Data)",
            "sharaxaad": "Marka maalmo ama toddobaadyo dheeraad ah oo xog ah laga soo ururiyo Open-Meteo, dib-u-tababarku wuxuu siinayaa model-ka fursad uu ku barto xaaladaha cimilada ugu dambeeya."
        },
        {
            "ciwaan": "Deganaanshaha iyo Ka Hortagga Khaladka (Continuous Reliability)",
            "sharaxaad": "Machine Learning ma aha wax hal mar la dhiso oo laga tago; nidaam xirfad leh wuxuu u baahan yahay kormeer joogto ah (MLOps) si loo hubiyo in accuracy-gu uusan hoos u dhicin waqti ka dib."
        }
    ]
    
    return {
        "active_model": active,
        "total_predictions_count": len(predictions),
        "evaluated_predictions_count": len(evaluated_predictions),
        "current_live_mae": current_live_mae,
        "drift_threshold": DRIFT_THRESHOLD_MAE,
        "drift_alert": drift_alert,
        "alert_message": alert_message,
        "recent_predictions": predictions[:15],
        "training_history": training_history,
        "educational_reasons": educational_reasons
    }


def simulate_real_actual_comparison(prediction_id: int, actual_value: float) -> Dict[str, Any]:
    """
    Allows user or teacher to provide the actual measured ground truth for a prediction
    to evaluate model drift and accuracy in real-time.
    """
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions_log WHERE id = ?;", (prediction_id,))
    pred_row = cursor.fetchone()
    
    if not pred_row:
        conn.close()
        return {"success": False, "error": f"Lama helin saadaal tirsigeedu yahay {prediction_id}"}
    
    pred = dict(zip(PRED_COLS, pred_row))
    
    abs_err = round(abs(float(actual_value) - float(pred["predicted_next_temp"])), 2)
    
    cursor.execute("""
        UPDATE predictions_log
        SET actual_next_temp = ?, absolute_error = ?, status = 'evaluated'
        WHERE id = ?;
    """, (actual_value, abs_err, prediction_id))
    
    conn.commit()
    conn.close()
    
    return {
        "success": True,
        "prediction_id": prediction_id,
        "predicted": pred["predicted_next_temp"],
        "actual": actual_value,
        "absolute_error": abs_err,
        "message": f"Cabbirka dhabta ah ({actual_value}°C) waa la kaydiyey. Qaladku waa {abs_err}°C."
    }
