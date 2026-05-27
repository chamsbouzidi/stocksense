"""
main.py — API FastAPI de serving du modèle StockSense
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
from datetime import date as date_type

# ── Chargement du modèle au démarrage ──────────────────
MODEL_PATH = os.getenv('MODEL_PATH', 'models/model.pkl')

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Modèle chargé depuis {MODEL_PATH}")
except FileNotFoundError:
    raise RuntimeError(f"Modèle introuvable : {MODEL_PATH}")

# ── Application ────────────────────────────────────────
app = FastAPI(
    title="StockSense API",
    description="API de prévision de la demande pour le retail mode.",
    version="1.0.0"
)

# ── Schéma de la requête ───────────────────────────────
class PredictionRequest(BaseModel):
    store_id:             int      = Field(..., ge=1, le=1115)
    prediction_date:      date_type = Field(...)
    promo:                int      = Field(..., ge=0, le=1)
    state_holiday:        int      = Field(0,  ge=0, le=3)
    school_holiday:       int      = Field(0,  ge=0, le=1)
    store_type:           int      = Field(..., ge=0, le=3)
    assortment:           int      = Field(..., ge=0, le=2)
    competition_distance: float    = Field(..., gt=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "store_id": 1,
                "prediction_date": "2025-01-15",
                "promo": 1,
                "state_holiday": 0,
                "school_holiday": 0,
                "store_type": 0,
                "assortment": 0,
                "competition_distance": 1270.0
            }
        }
    }

# ── Schéma de la réponse ───────────────────────────────
class PredictionResponse(BaseModel):
    store_id:        int
    prediction_date: str
    predicted_sales: float
    promo_active:    bool
    season:          str

# ── Endpoints ─────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "online",
        "service": "StockSense Demand Forecasting API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest):
    d           = request.prediction_date
    month       = d.month
    day_of_week = d.weekday()

    # Features métier — même logique que le notebook
    is_christmas = 1 if month in [11, 12] else 0
    is_summer    = 1 if month == 7        else 0
    is_winter    = 1 if month == 1        else 0
    is_sales     = 1 if month in [1, 7]   else 0
    is_monday    = 1 if day_of_week == 0  else 0
    is_saturday  = 1 if day_of_week == 5  else 0

    features = np.array([[
        request.store_id,
        day_of_week,
        request.promo,
        request.state_holiday,
        request.school_holiday,
        request.store_type,
        request.assortment,
        request.competition_distance,
        d.year,
        month,
        d.isocalendar()[1],
        d.day,
        is_christmas,
        is_summer,
        is_winter,
        is_sales,
        is_monday,
        is_saturday
    ]])

    try:
        prediction = float(model.predict(features)[0])
        prediction = max(0, round(prediction, 2))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur : {str(e)}")

    season_map = {
        12: "Hiver — Pic Noël",    1: "Hiver — Soldes",
        2:  "Hiver",               3: "Printemps",
        4:  "Printemps",           5: "Printemps",
        6:  "Été",                 7: "Été — Soldes",
        8:  "Été",                 9: "Automne",
        10: "Automne",            11: "Automne — Pré-Noël"
    }

    return PredictionResponse(
        store_id=request.store_id,
        prediction_date=str(request.prediction_date),
        predicted_sales=prediction,
        promo_active=bool(request.promo),
        season=season_map.get(month, "Inconnu")
    )