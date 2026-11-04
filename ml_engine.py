import os
import pickle
import json
import sqlite3
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
from database import get_db, update_step_status

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLS = ["temperature", "relative_humidity", "surface_pressure", "wind_speed"]

MODEL_EXPLANATIONS = {
    "baseline": {
        "name_somali": "Persistence Baseline (Heerkulkii Hore)",
        "english": "Persistence Baseline Model",
        "description": "Waa model aasaasi ah oo qiyaasaya in heerkulka saacadda xigta uu la mid noqon doono heerkulka hadda jira (Next Temp = Current Temp). Cimilada gudaheeda, heerkulku si lama-filaan ah kuma beddelmo 1 saac gudaheed.",
        "pros": "Aad u fudud, uma baahna tababar ama awood computer, wuxuuna noqdaa bartilmaameedka (benchmark) lagu cabbiro in moodallada kale wax tarayaan iyo in kale.",
        "cons": "Ma awoodo inuu barto isbeddelka qoyaanka, cadaadiska ama dabaysha, mana saadaalin karo marka cimiladu si degdeg ah isu beddesho (sida roob ama qorrax soo bax)."
    },
    "linear_regression": {
        "name_somali": "Linear Regression (Xiriirka Toosan)",
        "english": "Linear Regression",
        "description": "Waa algorithm xisaabeed oo raadinaya xiriir toosan (linear relationship) oo u dhexeeya heerkulka, qoyaanka, dabaysha iyo cadaadiska hawada.",
        "pros": "Aad u dheereeya, si sahlan loo fahmi karo (interpretable), wuxuuna si fiican u shaqeeyaa marka xiriirku toosan yahay.",
        "cons": "Ma qaban karo xiriirrada qallafsan ee qaloocan (non-linear patterns) ee cimilada dabiiciga ah leedahay."
    },
    "random_forest": {
        "name_somali": "Random Forest Regressor (Kaynta Go'aannada)",
        "english": "Random Forest Regressor",
        "description": "Waa algorithm awood badan oo ka kooban geedo go'aan badan (ensemble of decision trees). Geed kasta wuxuu xogta ka eegaa xagal kala duwan, natiijooyinka la isku darayna waxay bixiyaan saadaal sax ah.",
        "pros": "Wuxuu qabtaa qaababka qallafsan ee cimilada (non-linear interactions), aad buu u adkaystaa buuqa (noise), badanaana wuxuu keenaa qaladka ugu yar (lowest MAE/RMSE).",
        "cons": "Wuxuu u baahan yahay xusuus iyo tababar dheeraad ah, wuxuuna ka mugdi badan yahay Linear Regression (Black-box nature)."
    }
}


class BaselinePersistenceModel:
    """Predicts next hour temperature equals current temperature"""
    def __init__(self):
        self.is_fitted = False
        
    def fit(self, X, y=None):
        self.is_fitted = True
        return self
        
    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            return np.array([X[0]])
        return X[:, 0]


class PureLinearRegression:
    """Ordinary Least Squares Linear Regression using numpy.linalg.lstsq"""
    def __init__(self):
        self.intercept = 0.0
        self.weights = None
        self.is_fitted = False

    def fit(self, X, y):
        X_mat = np.asarray(X, dtype=np.float64)
        y_vec = np.asarray(y, dtype=np.float64)
        X_bias = np.c_[np.ones(X_mat.shape[0]), X_mat]
        beta, _, _, _ = np.linalg.lstsq(X_bias, y_vec, rcond=None)
        self.intercept = float(beta[0])
        self.weights = beta[1:]
        self.is_fitted = True
        return self

    def predict(self, X):
        X_mat = np.asarray(X, dtype=np.float64)
        if X_mat.ndim == 1:
            X_mat = X_mat.reshape(1, -1)
        return self.intercept + np.dot(X_mat, self.weights)


class DecisionTreeRegressor:
    """Single Decision Tree for Regression"""
    def __init__(self, max_depth=5, min_samples_split=6):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y, depth=0):
        if len(y) < self.min_samples_split or depth >= self.max_depth or np.var(y) < 1e-4:
            return {"val": float(np.mean(y))}
        
        n_features = X.shape[1]
        best_feat, best_thresh, best_var_red = None, None, -1.0
        current_var = np.var(y) * len(y)

        for feat in range(n_features):
            values = np.unique(X[:, feat])
            if len(values) > 12:
                thresholds = np.percentile(values, np.linspace(10, 90, 8))
            else:
                thresholds = (values[:-1] + values[1:]) / 2.0
            
            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask
                if not np.any(left_mask) or not np.any(right_mask):
                    continue
                var_left = np.var(y[left_mask]) * np.sum(left_mask)
                var_right = np.var(y[right_mask]) * np.sum(right_mask)
                var_red = current_var - (var_left + var_right)
                if var_red > best_var_red:
                    best_var_red = var_red
                    best_feat = feat
                    best_thresh = float(thresh)

        if best_feat is None:
            return {"val": float(np.mean(y))}

        left_mask = X[:, best_feat] <= best_thresh
        return {
            "feat": best_feat,
            "thresh": best_thresh,
            "left": self.fit(X[left_mask], y[left_mask], depth + 1),
            "right": self.fit(X[~left_mask], y[~left_mask], depth + 1)
        }

    def _predict_row(self, row, node):
        if "val" in node:
            return node["val"]
        if row[node["feat"]] <= node["thresh"]:
            return self._predict_row(row, node["left"])
        return self._predict_row(row, node["right"])

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            return np.array([self._predict_row(X, self.tree)])
        return np.array([self._predict_row(r, self.tree) for r in X])


class PureRandomForestRegressor:
    """Ensemble of Decision Trees with bootstrap bagging"""
    def __init__(self, n_estimators=10, max_depth=6, min_samples_split=5, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []
        self.is_fitted = False

    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples = len(X)
        self.trees = []
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.tree = tree.fit(X[indices], y[indices])
            self.trees.append(tree)
        self.is_fitted = True
        return self

    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(preds, axis=0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 4)
    }


def _load_cleaned_data(conn) -> tuple:
    """Load cleaned_weather as numpy arrays. Returns (recorded_at_list, X, y)."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT recorded_at, temperature, relative_humidity, surface_pressure, wind_speed, next_hour_temp
        FROM cleaned_weather
        ORDER BY recorded_at ASC;
    """)
    rows = cursor.fetchall()
    
    if not rows:
        return [], None, None
    
    recorded_at = [r[0] for r in rows]
    X = np.array([[r[1], r[2], r[3], r[4]] for r in rows], dtype=np.float64)
    y = np.array([r[5] for r in rows], dtype=np.float64)
    return recorded_at, X, y


def train_and_evaluate_models() -> Dict[str, Any]:
    conn = get_db()
    recorded_at, X, y = _load_cleaned_data(conn)
    conn.close()
    
    if X is None or len(X) == 0:
        return {
            "success": False,
            "error": "Ma jirto xog nadiif ah oo la tababari karo. Fadlan marka hore run-garee Tallaabada 2 (Nadiifinta)."
        }
    
    total_samples = len(X)
    if total_samples < 30:
        return {
            "success": False,
            "error": f"Xogta nadiifka ah aad bay u yar tahay ({total_samples} saf). Fadlan marka hore ururi xog ku filan (ugu yaraan 30-50 saf) Tallaabada 1."
        }
    
    warning_msg = None
    if total_samples < 100:
        warning_msg = f"Digniin: Dataset-ku wuxuu ka kooban yahay {total_samples} saf oo keliya. Saadaasha Machine Learning-ku waxay noqon kartaa mid xaddidan ilaa xog badan la ururiyo."
    
    # Chronological Split: 70% Train, 15% Validation, 15% Test
    n_train = int(total_samples * 0.70)
    n_val = int(total_samples * 0.15)
    n_test = total_samples - n_train - n_val
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
    
    models_to_train = {
        "baseline": BaselinePersistenceModel(),
        "linear_regression": PureLinearRegression(),
        "random_forest": PureRandomForestRegressor(n_estimators=10, max_depth=6, random_state=42)
    }
    
    results = {}
    best_model_key = None
    best_val_mae = float("inf")
    
    for key, model in models_to_train.items():
        model.fit(X_train, y_train)
        
        val_pred = model.predict(X_val)
        val_metrics = compute_metrics(y_val, val_pred)
        
        test_pred = model.predict(X_test)
        test_metrics = compute_metrics(y_test, test_pred)
        
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_model_key = key
        
        results[key] = {
            "model_key": key,
            "name_somali": MODEL_EXPLANATIONS[key]["name_somali"],
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "test_predictions": [round(float(p), 2) for p in test_pred],
            "model_instance": model
        }
    
    # Save the best model using pure python pickle
    best_model_info = results[best_model_key]
    saved_model_path = os.path.join(MODELS_DIR, f"best_weather_model_{best_model_key}.pkl")
    with open(saved_model_path, "wb") as f:
        pickle.dump(best_model_info["model_instance"], f)
    
    # Save metadata to models_registry
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("UPDATE models_registry SET is_active = 0 WHERE is_active = 1;")
    
    trained_at = datetime.utcnow().isoformat()
    for key, info in results.items():
        is_active = 1 if key == best_model_key else 0
        cursor.execute("""
        INSERT INTO models_registry (
            model_name, algorithm, hyperparameters, train_samples, val_samples, test_samples,
            val_mae, val_rmse, val_r2, test_mae, test_rmse, test_r2, model_file_path, is_active, trained_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            MODEL_EXPLANATIONS[key]["name_somali"],
            key,
            json.dumps({"features": FEATURE_COLS}),
            n_train,
            n_val,
            n_test,
            info["val_metrics"]["mae"],
            info["val_metrics"]["rmse"],
            info["val_metrics"]["r2"],
            info["test_metrics"]["mae"],
            info["test_metrics"]["rmse"],
            info["test_metrics"]["r2"],
            saved_model_path if is_active else None,
            is_active,
            trained_at
        ))
    
    conn.commit()
    conn.close()
    
    # Update status for Steps 3, 4, 5, 6
    update_step_status(3, "completed", f"Model-ka ugu fiican waxaa loo doortay '{MODEL_EXPLANATIONS[best_model_key]['name_somali']}' sababtoo ah wuxuu keenay Validation MAE ugu yar ({best_val_mae}°C).")
    update_step_status(4, "completed", f"Tababarku waa guulaystay: {n_train} saf oo tababar ah, {n_val} validation, {n_test} test.")
    update_step_status(5, "completed", f"Qiimaynta Test-ka: MAE={results[best_model_key]['test_metrics']['mae']}°C, RMSE={results[best_model_key]['test_metrics']['rmse']}°C, R²={results[best_model_key]['test_metrics']['r2']}.")
    update_step_status(6, "completed", f"Model-ka firfircoon ({best_model_key}) wuxuu diyaar u yahay saadaasha tooska ah.")
    
    # Build chart comparison data (actual vs predicted vs baseline)
    test_dates = recorded_at[n_train + n_val:]
    actual_test_temps = y_test.tolist()
    best_pred_temps = results[best_model_key]["test_predictions"]
    baseline_pred_temps = results["baseline"]["test_predictions"]
    
    chart_comparison = []
    step_skip = max(1, len(test_dates) // 48)
    for i in range(0, len(test_dates), step_skip):
        chart_comparison.append({
            "time": test_dates[i].replace("T", " "),
            "actual": round(actual_test_temps[i], 1),
            "predicted": round(best_pred_temps[i], 1),
            "baseline": round(baseline_pred_temps[i], 1),
            "error": round(abs(actual_test_temps[i] - best_pred_temps[i]), 2)
        })
    
    return {
        "success": True,
        "message": f"Tababarka iyo qiimayntu waxay ku dhammaadeen guul. Model-ka ku guulaystay waa '{MODEL_EXPLANATIONS[best_model_key]['name_somali']}'.",
        "best_model": best_model_key,
        "best_model_name": MODEL_EXPLANATIONS[best_model_key]["name_somali"],
        "splits": {
            "total": total_samples,
            "train": n_train,
            "val": n_val,
            "test": n_test
        },
        "warning": warning_msg,
        "models_summary": {
            k: {
                "name": MODEL_EXPLANATIONS[k]["name_somali"],
                "val": results[k]["val_metrics"],
                "test": results[k]["test_metrics"]
            }
            for k in results
        },
        "chart_data": chart_comparison
    }


def predict_single(temperature: float, relative_humidity: float, surface_pressure: float, wind_speed: float) -> Dict[str, Any]:
    """
    Tallaabada 6: Deploy the Model (Saadaal Toos ah)
    """
    errors = []
    if not (15.0 <= temperature <= 50.0):
        errors.append(f"Heerkulka ({temperature}°C) wuxuu ka baxsan yahay xadka macquulka ah ee Muqdisho (15°C - 50°C).")
    if not (10.0 <= relative_humidity <= 100.0):
        errors.append(f"Qoyaanka ({relative_humidity}%) waa inuu u dhexeeyaa 10% iyo 100%.")
    if not (970.0 <= surface_pressure <= 1045.0):
        errors.append(f"Cadaadiska ({surface_pressure} hPa) waa inuu u dhexeeyaa 970 iyo 1045 hPa.")
    if not (0.0 <= wind_speed <= 100.0):
        errors.append(f"Xawaaraha dabaysha ({wind_speed} km/h) waa inuu u dhexeeyaa 0 iyo 100 km/h.")
    
    if errors:
        return {"success": False, "errors": errors}
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM models_registry WHERE is_active = 1 ORDER BY id DESC LIMIT 1;")
    active_row = cursor.fetchone()
    
    reg_cols = ["id", "model_name", "algorithm", "hyperparameters", "train_samples", "val_samples", "test_samples",
                "val_mae", "val_rmse", "val_r2", "test_mae", "test_rmse", "test_r2", "model_file_path", "is_active", "trained_at"]
    
    if not active_row:
        conn.close()
        return {
            "success": False,
            "error": "Ma jiro model firfircoon oo la tababaray oo diyaar ah. Fadlan marka hore tababar model-ka Tallaabada 4."
        }
    
    active = dict(zip(reg_cols, active_row))
    model_path = active.get("model_file_path")
    
    if not model_path or not os.path.exists(model_path):
        conn.close()
        return {
            "success": False,
            "error": "Ma jiro model file diyaar ah. Fadlan tababar model-ka Tallaabada 4."
        }
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    X_input = np.array([[temperature, relative_humidity, surface_pressure, wind_speed]])
    pred_res = model.predict(X_input)
    pred_val = float(pred_res[0]) if hasattr(pred_res, "__len__") else float(pred_res)
    pred_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Log prediction
    cursor.execute("""
    INSERT INTO predictions_log (
        model_id, model_algorithm, input_temperature, input_humidity,
        input_wind_speed, input_surface_pressure, predicted_next_temp, prediction_time
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
    """, (
        active["id"],
        active["algorithm"],
        temperature,
        relative_humidity,
        wind_speed,
        surface_pressure,
        round(pred_val, 2),
        pred_time
    ))
    conn.commit()
    conn.close()
    
    temp_diff = round(pred_val - temperature, 2)
    direction_so = "kor buu u kici doonaa" if temp_diff > 0 else "hoos buu u dhici doonaa" if temp_diff < 0 else "isbeddel ma samayn doono"
    
    return {
        "success": True,
        "predicted_next_hour_temp": round(pred_val, 2),
        "temperature_change": temp_diff,
        "direction_explanation": f"Heerkulku saacadda xigta {direction_so} ({abs(temp_diff)}°C).",
        "unit": "°C",
        "model_used": active["model_name"],
        "algorithm": active["algorithm"],
        "prediction_time": pred_time,
        "inputs": {
            "temperature": f"{temperature} °C",
            "relative_humidity": f"{relative_humidity} %",
            "surface_pressure": f"{surface_pressure} hPa",
            "wind_speed": f"{wind_speed} km/h"
        }
    }


def get_active_model_details() -> Dict[str, Any]:
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM models_registry WHERE is_active = 1 ORDER BY id DESC LIMIT 1;")
    active_row = cursor.fetchone()
    
    cursor.execute("SELECT * FROM models_registry ORDER BY id DESC;")
    all_rows = cursor.fetchall()
    conn.close()
    
    reg_cols = ["id", "model_name", "algorithm", "hyperparameters", "train_samples", "val_samples", "test_samples",
                "val_mae", "val_rmse", "val_r2", "test_mae", "test_rmse", "test_r2", "model_file_path", "is_active", "trained_at"]
    
    return {
        "active_model": dict(zip(reg_cols, active_row)) if active_row else None,
        "history": [dict(zip(reg_cols, r)) for r in all_rows],
        "explanations": MODEL_EXPLANATIONS
    }
