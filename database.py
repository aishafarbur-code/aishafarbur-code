import sqlite3
import os
import json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weather_station.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()

    # 1. Raw weather data collected from Open-Meteo
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS raw_weather (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recorded_at TEXT NOT NULL UNIQUE,
        temperature REAL,
        relative_humidity REAL,
        surface_pressure REAL,
        wind_speed REAL,
        next_hour_temp REAL,
        data_type TEXT DEFAULT 'historical',
        collected_at TEXT NOT NULL
    );
    """)

    # 2. Cleaning and preparation audit logs
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS clean_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_at TEXT NOT NULL,
        total_raw INTEGER NOT NULL,
        duplicates_removed INTEGER NOT NULL,
        missing_handled INTEGER NOT NULL,
        outliers_adjusted INTEGER NOT NULL,
        final_clean_count INTEGER NOT NULL,
        details_json TEXT
    );
    """)

    # 3. Cleaned dataset table (persisted so that tababarka uses clean data)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cleaned_weather (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recorded_at TEXT NOT NULL UNIQUE,
        temperature REAL NOT NULL,
        relative_humidity REAL NOT NULL,
        surface_pressure REAL NOT NULL,
        wind_speed REAL NOT NULL,
        next_hour_temp REAL NOT NULL
    );
    """)

    # 4. Trained models registry
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS models_registry (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        algorithm TEXT NOT NULL,
        hyperparameters TEXT,
        train_samples INTEGER NOT NULL,
        val_samples INTEGER NOT NULL,
        test_samples INTEGER NOT NULL,
        val_mae REAL,
        val_rmse REAL,
        val_r2 REAL,
        test_mae REAL,
        test_rmse REAL,
        test_r2 REAL,
        model_file_path TEXT,
        is_active INTEGER DEFAULT 0,
        trained_at TEXT NOT NULL
    );
    """)

    # 5. Live predictions history and monitoring
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_id INTEGER,
        model_algorithm TEXT NOT NULL,
        input_temperature REAL NOT NULL,
        input_humidity REAL NOT NULL,
        input_wind_speed REAL NOT NULL,
        input_surface_pressure REAL NOT NULL,
        predicted_next_temp REAL NOT NULL,
        prediction_time TEXT NOT NULL,
        actual_next_temp REAL,
        absolute_error REAL,
        status TEXT DEFAULT 'pending'
    );
    """)

    # 6. Step status tracker (Lama bilaabin, Socota, Dhammaystiran)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS steps_status (
        step_id INTEGER PRIMARY KEY,
        step_code TEXT UNIQUE NOT NULL,
        step_name TEXT NOT NULL,
        status TEXT NOT NULL, -- 'not_started', 'in_progress', 'completed'
        updated_at TEXT NOT NULL,
        notes TEXT
    );
    """)

    # Initialize default 7 steps if not exist
    default_steps = [
        (1, "collect_data", "Tallaabada 1: Collect Data (Ururinta Xogta)", "not_started", "Xogta cimilada Muqdisho ee Open-Meteo"),
        (2, "prepare_clean", "Tallaabada 2: Prepare & Clean Data (Diyaarinta & Nadiifinta)", "not_started", "Nadiifinta missing values, outliers iyo formatting"),
        (3, "choose_model", "Tallaabada 3: Choose a Model (Doorashada Model-ka)", "not_started", "Baseline, Linear Regression, Random Forest"),
        (4, "train_model", "Tallaabada 4: Train the Model (Tababarka Model-ka)", "not_started", "Time-series split iyo scikit-learn"),
        (5, "evaluate_model", "Tallaabada 5: Test & Evaluate (Tijaabinta & Qiimaynta)", "not_started", "MAE, RMSE, R² iyo charts"),
        (6, "deploy_model", "Tallaabada 6: Deploy the Model (Gelinta Nidaamka Saadaasha)", "not_started", "FastAPI inference endpoint & form"),
        (7, "monitor_improve", "Tallaabada 7: Monitor & Improve (La Socodka & Horumarinta)", "not_started", "Dabagalka saxnaanta iyo dib-u-tababar")
    ]

    for step in default_steps:
        cursor.execute("""
        INSERT OR IGNORE INTO steps_status (step_id, step_code, step_name, status, updated_at, notes)
        VALUES (?, ?, ?, ?, ?, ?);
        """, (step[0], step[1], step[2], step[3], datetime.utcnow().isoformat(), step[4]))

    conn.commit()
    conn.close()

def update_step_status(step_id: int, status: str, notes: str = None):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
    UPDATE steps_status 
    SET status = ?, updated_at = ?, notes = COALESCE(?, notes)
    WHERE step_id = ?;
    """, (status, datetime.utcnow().isoformat(), notes, step_id))
    conn.commit()
    conn.close()

def get_all_steps_status():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM steps_status ORDER BY step_id ASC;")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]
