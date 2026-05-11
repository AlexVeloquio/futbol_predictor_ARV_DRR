"""
app.py — API local para el predictor de Liga MX.

Corre con: uvicorn app:app --reload
Abre en:   http://localhost:8000
"""

import os
import sys
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Agregar src/inference al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "inference"))
from predict import Predictor

app = FastAPI(title="Predictor Liga MX")

# Cargar modelo al iniciar
predictor = Predictor("models")


class PredictRequest(BaseModel):
    home_team: str
    away_team: str


@app.get("/api/teams")
def get_teams():
    """Retorna la lista de equipos disponibles."""
    return {"teams": predictor.get_teams()}


@app.post("/api/predict")
def predict_match(req: PredictRequest):
    """Predice el resultado de un partido."""
    try:
        result = predictor.predict(req.home_team, req.away_team)
        return result
    except ValueError as e:
        return {"error": str(e)}


# Servir frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")
