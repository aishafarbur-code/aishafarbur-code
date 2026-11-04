import sqlite3
import numpy as np
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from database import get_db, update_step_status

# Xuduudaha macquulka ah ee cimilada Muqdisho (Realistic Meteorological Boundaries)
BOUNDS = {
    "temperature": {"min": 15.0, "max": 48.0, "unit": "°C"},
    "relative_humidity": {"min": 10.0, "max": 100.0, "unit": "%"},
    "surface_pressure": {"min": 980.0, "max": 1040.0, "unit": "hPa"},
    "wind_speed": {"min": 0.0, "max": 90.0, "unit": "km/h"}
}

FEATURE_COLS = ["temperature", "relative_humidity", "surface_pressure", "wind_speed"]

def _load_raw_data(conn) -> List[Dict]:
    """Load raw_weather table as list of dicts, ordered by recorded_at ASC."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT recorded_at, temperature, relative_humidity, surface_pressure, wind_speed, next_hour_temp
        FROM raw_weather
        ORDER BY recorded_at ASC;
    """)
    rows = cursor.fetchall()
    cols = ["recorded_at", "temperature", "relative_humidity", "surface_pressure", "wind_speed", "next_hour_temp"]
    return [dict(zip(cols, r)) for r in rows]

def _linear_interpolate_col(values: List[Optional[float]]) -> List[float]:
    """Linear interpolation for a list of float values (None = missing)."""
    n = len(values)
    result = list(values)
    
    # Forward fill for leading NaN
    first_valid = None
    for i in range(n):
        if result[i] is not None:
            first_valid = result[i]
            break
    if first_valid is None:
        return [0.0] * n  # all missing
    for i in range(n):
        if result[i] is None:
            result[i] = first_valid
        else:
            first_valid = result[i]
            break

    # Linear interpolation between valid values
    i = 0
    while i < n:
        if result[i] is None:
            # Find next valid
            j = i + 1
            while j < n and result[j] is None:
                j += 1
            if j < n:
                # Interpolate between i-1 and j
                start_val = result[i - 1] if i > 0 else result[j]
                end_val = result[j]
                for k in range(i, j):
                    t = (k - (i - 1)) / (j - (i - 1))
                    result[k] = start_val + t * (end_val - start_val)
            else:
                # Back fill
                fill_val = result[i - 1] if i > 0 else 0.0
                for k in range(i, n):
                    result[k] = fill_val
        i += 1

    # Back fill any remaining None at end
    last_valid = None
    for i in range(n - 1, -1, -1):
        if result[i] is not None:
            last_valid = result[i]
            break
    if last_valid is not None:
        for i in range(n):
            if result[i] is None:
                result[i] = last_valid

    return [v if v is not None else 0.0 for v in result]

def clean_and_prepare_dataset() -> Dict[str, Any]:
    """
    Tallaabada 2: Prepare & Clean Data (Diyaarinta iyo Nadiifinta Xogta)
    """
    conn = get_db()
    rows = _load_raw_data(conn)
    total_raw_rows = len(rows)
    
    if total_raw_rows == 0:
        conn.close()
        return {
            "success": False,
            "error": "Ma jirto xog lagu nadiifiyo database-ka. Fadlan marka hore ku dhufo badhanka 'Ururi Xogta' ee Tallaabada 1."
        }
    
    audit_log = []
    
    # 2. Check duplicates — keep first occurrence of each recorded_at
    seen_times = set()
    unique_rows = []
    duplicate_count = 0
    for r in rows:
        if r["recorded_at"] in seen_times:
            duplicate_count += 1
        else:
            seen_times.add(r["recorded_at"])
            unique_rows.append(r)
    
    if duplicate_count > 0:
        audit_log.append(f"Waxaa laga saaray {duplicate_count} saf oo xog isku mid ah (Duplicates) oo taariikhda ku celceliyey.")
    else:
        audit_log.append("Lama helin wax safaf isku mid ah (0 Duplicates found).")
    
    rows = unique_rows
    n = len(rows)
    
    # 3. Missing values analysis before cleaning
    missing_before = {}
    for col in FEATURE_COLS + ["next_hour_temp"]:
        missing_before[col] = sum(1 for r in rows if r.get(col) is None)
    
    # 4. Outlier detection and clipping for feature cols
    outliers_detected = {}
    outliers_adjusted_count = 0
    for col, b in BOUNDS.items():
        cnt = 0
        for r in rows:
            v = r.get(col)
            if v is not None and (v < b["min"] or v > b["max"]):
                cnt += 1
        outliers_detected[col] = cnt
        if cnt > 0:
            outliers_adjusted_count += cnt
            for r in rows:
                v = r.get(col)
                if v is not None:
                    r[col] = max(b["min"], min(b["max"], v))
            audit_log.append(f"Tiirka '{col}' waxaa lagu hagaajiyey {cnt} qiime oo xadka dhaafay (Clipped to realistic meteorological bounds).")
    
    # 5. Time-series interpolation for features
    missing_features_count = sum(missing_before.get(c, 0) for c in FEATURE_COLS)
    if missing_features_count > 0:
        for col in FEATURE_COLS:
            vals = [r.get(col) for r in rows]
            interp = _linear_interpolate_col(vals)
            for i, r in enumerate(rows):
                r[col] = interp[i]
        audit_log.append(f"Waxaa lagu xalliyey {missing_features_count} qiime oo ka maqnaa features-ka habka Linear Interpolation oo u gaar ah xogta waqtiga (Time-series continuity).")
    else:
        audit_log.append("Features-ka (heerkulka, qoyaanka, cadaadiska, dabaysha) ma lahayn wax qiimo maqan ah.")
    
    # 6. Drop rows where next_hour_temp is None (last row or missing target)
    rows_before_target_drop = len(rows)
    rows = [r for r in rows if r.get("next_hour_temp") is not None]
    rows_dropped_target = rows_before_target_drop - len(rows)
    if rows_dropped_target > 0:
        audit_log.append(f"Waxaa laga saaray {rows_dropped_target} saf (sida saacadda ugu dambeysa) sababtoo ah target-ka heerkulka saacadda xigta lama hayo weli.")
    
    final_clean_count = len(rows)
    
    # 7. Save clean data into cleaned_weather table
    cursor = conn.cursor()
    cursor.execute("DELETE FROM cleaned_weather;")
    
    for r in rows:
        cursor.execute("""
        INSERT INTO cleaned_weather (recorded_at, temperature, relative_humidity, surface_pressure, wind_speed, next_hour_temp)
        VALUES (?, ?, ?, ?, ?, ?);
        """, (
            r["recorded_at"],
            float(r["temperature"]),
            float(r["relative_humidity"]),
            float(r["surface_pressure"]),
            float(r["wind_speed"]),
            float(r["next_hour_temp"])
        ))
    
    # Save clean run audit
    details = {
        "missing_before": missing_before,
        "outliers_detected": outliers_detected,
        "audit_log": audit_log,
        "features": FEATURE_COLS,
        "target": "next_hour_temp",
        "data_leakage_protection": "Waqtiga ayaa loo kala horreeyay (Chronological sorting). Random shuffle lama isticmaalin si xogta mustaqbalku aysan ugu dhex dhuuman xogta hore (Data Leakage Prevention)."
    }
    
    cursor.execute("""
    INSERT INTO clean_runs (run_at, total_raw, duplicates_removed, missing_handled, outliers_adjusted, final_clean_count, details_json)
    VALUES (?, ?, ?, ?, ?, ?, ?);
    """, (
        datetime.utcnow().isoformat(),
        total_raw_rows,
        duplicate_count,
        missing_features_count,
        outliers_adjusted_count,
        final_clean_count,
        json.dumps(details, ensure_ascii=False)
    ))
    conn.commit()
    conn.close()
    
    # Update Step 2 Status
    update_step_status(2, "completed", f"Nadiifinta waa la dhammaystiray: {final_clean_count} saf oo nadiif ah ayaa diyaar u ah tababarka.")
    
    # Preview: first 5 rows
    preview_sample = rows[:5] if rows else []
    
    return {
        "success": True,
        "message": "Diyaarinta iyo Nadiifinta xogta si guul leh ayaa loo dhammeeyay.",
        "summary": {
            "total_raw": total_raw_rows,
            "duplicates_removed": duplicate_count,
            "missing_features_handled": missing_features_count,
            "outliers_adjusted": outliers_adjusted_count,
            "final_clean_count": final_clean_count,
            "dropped_incomplete_rows": rows_dropped_target
        },
        "audit_log": audit_log,
        "preview_sample": preview_sample,
        "features": FEATURE_COLS,
        "target": "next_hour_temp"
    }


def get_latest_cleaning_status() -> Dict[str, Any]:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM clean_runs ORDER BY id DESC LIMIT 1;")
    row = cursor.fetchone()
    
    cursor.execute("SELECT COUNT(*) FROM cleaned_weather;")
    clean_count = cursor.fetchone()[0]
    conn.close()
    
    if not row:
        return {"has_cleaned_data": False, "clean_count": 0}
    
    # clean_runs columns: id, run_at, total_raw, duplicates_removed, missing_handled, outliers_adjusted, final_clean_count, details_json
    cols = ["id", "run_at", "total_raw", "duplicates_removed", "missing_handled", "outliers_adjusted", "final_clean_count", "details_json"]
    row_dict = dict(zip(cols, row))
    details = json.loads(row_dict.get("details_json", "{}"))
    
    return {
        "has_cleaned_data": True,
        "clean_count": clean_count,
        "run_at": row_dict["run_at"],
        "total_raw": row_dict["total_raw"],
        "duplicates_removed": row_dict["duplicates_removed"],
        "missing_handled": row_dict["missing_handled"],
        "outliers_adjusted": row_dict["outliers_adjusted"],
        "final_clean_count": row_dict["final_clean_count"],
        "details": details
    }
