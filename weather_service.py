import httpx
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from database import get_db, update_step_status

# Muqdisho, Soomaaliya coordinates
MOGADISHU_LAT = 2.0469
MOGADISHU_LON = 45.3182

async def fetch_current_weather_api() -> Dict[str, Any]:
    """
    Ka soo qaad xogta cimilada hadda jirta ee Muqdisho Open-Meteo API.
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={MOGADISHU_LAT}&longitude={MOGADISHU_LON}&"
        f"current=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m&"
        f"timezone=Africa%2FMogadishu"
    )
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            
            curr = data.get("current", {})
            return {
                "success": True,
                "recorded_at": curr.get("time"),
                "temperature": curr.get("temperature_2m"),
                "relative_humidity": curr.get("relative_humidity_2m"),
                "surface_pressure": curr.get("surface_pressure"),
                "wind_speed": curr.get("wind_speed_10m"),
                "units": {
                    "temperature": "°C",
                    "relative_humidity": "%",
                    "surface_pressure": "hPa",
                    "wind_speed": "km/h"
                },
                "location": "Muqdisho, Soomaaliya (2.0469° N, 45.3182° E)"
            }
        except httpx.HTTPStatusError as e:
            return {"success": False, "error": f"Cilad API (HTTP {e.response.status_code}): {e.response.text}"}
        except httpx.RequestError as e:
            return {"success": False, "error": f"Xiriirka internet-ka ayaa go'an ama API-ga lama gaari karo: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": f"Khalad aan la filayn: {str(e)}"}

async def fetch_historical_and_forecast_data(past_days: int = 30) -> Dict[str, Any]:
    """
    Open-Meteo API waxay bixisaa xog taariikhi ah (past_days) iyo saadaal (forecast_days).
    Waxaan ka soo ururinaynaa xogta saacadlaha ah ee Muqdisho si aan ugu tababarno Machine Learning.
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={MOGADISHU_LAT}&longitude={MOGADISHU_LON}&"
        f"hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m&"
        f"past_days={past_days}&forecast_days=2&"
        f"timezone=Africa%2FMogadishu"
    )
    
    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            temps = hourly.get("temperature_2m", [])
            humidities = hourly.get("relative_humidity_2m", [])
            pressures = hourly.get("surface_pressure", [])
            winds = hourly.get("wind_speed_10m", [])
            
            if not times:
                return {"success": False, "error": "API-gu ma keenin wax xog saacadle ah."}
            
            # Compute target: next_hour_temp = temperature[i+1]
            # (shift(-1) in numpy style)
            n = len(times)
            
            # Determine data_type: historical vs current vs forecast
            now_iso = datetime.now().strftime("%Y-%m-%dT%H:00")
            
            def categorize(row_time: str) -> str:
                if row_time < now_iso:
                    return "historical"
                elif row_time == now_iso:
                    return "current"
                else:
                    return "forecast"
            
            # Save into raw_weather SQLite table
            conn = get_db()
            cursor = conn.cursor()
            
            inserted_count = 0
            collected_time = datetime.utcnow().isoformat()
            
            for i in range(n):
                t = times[i]
                temp = temps[i] if i < len(temps) else None
                hum = humidities[i] if i < len(humidities) else None
                pres = pressures[i] if i < len(pressures) else None
                wind = winds[i] if i < len(winds) else None
                
                # next_hour_temp: temperature of next row (i+1), None for last row
                if i + 1 < len(temps):
                    next_temp = temps[i + 1]
                else:
                    next_temp = None
                
                data_type = categorize(t)
                
                try:
                    cursor.execute("""
                    INSERT INTO raw_weather (
                        recorded_at, temperature, relative_humidity, surface_pressure, 
                        wind_speed, next_hour_temp, data_type, collected_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(recorded_at) DO UPDATE SET
                        temperature=excluded.temperature,
                        relative_humidity=excluded.relative_humidity,
                        surface_pressure=excluded.surface_pressure,
                        wind_speed=excluded.wind_speed,
                        next_hour_temp=excluded.next_hour_temp,
                        data_type=excluded.data_type,
                        collected_at=excluded.collected_at;
                    """, (
                        t,
                        temp,
                        hum,
                        pres,
                        wind,
                        next_temp,
                        data_type,
                        collected_time
                    ))
                    inserted_count += 1
                except Exception as e:
                    print("Row insertion issue:", e)
            
            conn.commit()
            
            # Count total rows in raw_weather
            cursor.execute("SELECT COUNT(*) FROM raw_weather")
            total_in_db = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM raw_weather WHERE data_type = 'historical'")
            hist_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM raw_weather WHERE data_type = 'forecast'")
            forecast_count = cursor.fetchone()[0]
            
            conn.close()
            
            # Update Step 1 Status to completed
            update_step_status(1, "completed", f"Waxaa la soo ururiyey {total_in_db} saf oo xog dhab ah.")
            
            return {
                "success": True,
                "message": f"Si guul leh ayaa loo soo ururiyey {n} saacadood oo xog ah.",
                "total_records": total_in_db,
                "historical_count": hist_count,
                "forecast_count": forecast_count,
                "date_range": {
                    "start": times[0] if times else None,
                    "end": times[-1] if times else None
                }
            }
            
        except httpx.HTTPStatusError as e:
            update_step_status(1, "in_progress", f"Khalad API: {e.response.status_code}")
            return {"success": False, "error": f"Cilad API (HTTP {e.response.status_code}): {e.response.text}"}
        except httpx.RequestError as e:
            update_step_status(1, "in_progress", "Khalad xiriirka internet-ka")
            return {"success": False, "error": f"Xiriirka internet-ka ayaa go'an ama API-ga lama gaari karo: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": f"Khalad lama filaan ah: {str(e)}"}

def get_raw_weather_records(limit: int = 200, offset: int = 0) -> Dict[str, Any]:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM raw_weather")
    total = cursor.fetchone()[0]
    
    cursor.execute("""
    SELECT recorded_at, temperature, relative_humidity, surface_pressure, wind_speed, next_hour_temp, data_type, collected_at
    FROM raw_weather
    ORDER BY recorded_at DESC
    LIMIT ? OFFSET ?
    """, (limit, offset))
    rows = cursor.fetchall()
    conn.close()
    
    columns = ["recorded_at", "temperature", "relative_humidity", "surface_pressure", "wind_speed", "next_hour_temp", "data_type", "collected_at"]
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "data": [dict(zip(columns, row)) for row in rows]
    }
