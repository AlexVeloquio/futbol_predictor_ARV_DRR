"""
predict.py — Inferencia: recibe dos equipos de Liga MX y predice el resultado.

Usa model.pkl + team_stats.json para construir el feature vector
a partir de los nombres de los equipos.

Uso local:
  from src.inference.predict import Predictor
  p = Predictor("models/")
  result = p.predict("América", "Guadalajara")

Uso en Lambda:
  Se cargan los artefactos desde /tmp/ (descargados de S3).
"""

import json
import os
import joblib
import numpy as np
from difflib import get_close_matches

LABELS = {0: "Home Win", 1: "Draw", 2: "Away Win"}
LABELS_ES = {0: "Victoria Local", 1: "Empate", 2: "Victoria Visitante"}

# Nombre canónico → alias comunes
TEAM_ALIASES = {
    "Club America": ["América", "America", "Club América", "Águilas"],
    "Guadalajara Chivas": ["Chivas", "Guadalajara", "Rebaño"],
    "Cruz Azul": ["Cruz Azul", "La Máquina", "Cementeros"],
    "U.N.A.M. - Pumas": ["Pumas", "UNAM", "Pumas UNAM"],
    "Tigres UANL": ["Tigres", "UANL"],
    "Monterrey": ["Rayados", "Monterrey"],
    "Santos Laguna": ["Santos", "Santos Laguna"],
    "Leon": ["León", "Leon", "La Fiera"],
    "Club Tijuana": ["Tijuana", "Xolos", "Club Tijuana"],
    "Toluca": ["Toluca", "Diablos Rojos"],
    "Pachuca": ["Pachuca", "Tuzos"],
    "Atlas": ["Atlas", "Rojinegros"],
    "Necaxa": ["Necaxa", "Rayos"],
    "Puebla": ["Puebla", "La Franja"],
    "Club Queretaro": ["Querétaro", "Queretaro", "Gallos"],
    "FC Juarez": ["Juárez", "Juarez", "FC Juárez", "Bravos"],
    "Atletico San Luis": ["San Luis", "Atlético San Luis", "Atletico San Luis"],
    "Mazatlán": ["Mazatlán", "Mazatlan", "Cañoneros"],
}

# Construir reverse lookup
_ALIAS_TO_CANONICAL = {}
for canonical, aliases in TEAM_ALIASES.items():
    _ALIAS_TO_CANONICAL[canonical.lower()] = canonical
    for alias in aliases:
        _ALIAS_TO_CANONICAL[alias.lower()] = canonical


def resolve_team_name(name: str, available_teams: list) -> str:
    """
    Resuelve el nombre de un equipo usando aliases y fuzzy matching.
    """
    # Intento 1: match exacto en aliases
    canonical = _ALIAS_TO_CANONICAL.get(name.lower())
    if canonical and canonical in available_teams:
        return canonical

    # Intento 2: match exacto directo
    if name in available_teams:
        return name

    # Intento 3: match case-insensitive
    name_lower = name.lower()
    for team in available_teams:
        if team.lower() == name_lower:
            return team

    # Intento 4: fuzzy matching
    matches = get_close_matches(name, available_teams, n=1, cutoff=0.5)
    if matches:
        return matches[0]

    return None


class Predictor:
    """Encapsula la carga del modelo y la predicción."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self._load_artifacts()

    def _load_artifacts(self):
        """Carga modelo, metadata, scaler y stats de equipos."""
        self.model = joblib.load(os.path.join(self.models_dir, "model.pkl"))

        with open(os.path.join(self.models_dir, "metadata.json"), "r") as f:
            self.metadata = json.load(f)

        with open(os.path.join(self.models_dir, "team_stats.json"), "r") as f:
            all_stats = json.load(f)

        self.team_stats = all_stats["teams"]
        self.h2h = all_stats.get("h2h", {})
        self.features = self.metadata["features"]
        self.teams = sorted(self.team_stats.keys())

        # Cargar scaler si el modelo lo requiere
        self.scaler = None
        if self.metadata.get("requires_scaling", False):
            scaler_path = os.path.join(self.models_dir, "scaler.pkl")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)

    def get_teams(self) -> list:
        """Retorna la lista de equipos disponibles."""
        return self.teams

    def predict(self, home_team: str, away_team: str) -> dict:
        """
        Predice el resultado de un partido.

        Args:
            home_team: nombre del equipo local (acepta aliases)
            away_team: nombre del equipo visitante (acepta aliases)

        Returns:
            dict con predicción, probabilidades y detalles.
        """
        # Resolver nombres
        home_resolved = resolve_team_name(home_team, self.teams)
        away_resolved = resolve_team_name(away_team, self.teams)

        if home_resolved is None:
            raise ValueError(
                f"Equipo local '{home_team}' no encontrado.\n"
                f"Equipos disponibles: {self.teams}"
            )
        if away_resolved is None:
            raise ValueError(
                f"Equipo visitante '{away_team}' no encontrado.\n"
                f"Equipos disponibles: {self.teams}"
            )

        if home_resolved == away_resolved:
            raise ValueError("El equipo local y visitante no pueden ser el mismo.")

        # Construir feature vector
        h = self.team_stats[home_resolved]
        a = self.team_stats[away_resolved]

        # H2H
        h2h_key = f"{home_resolved}_vs_{away_resolved}"
        h2h_score = self.h2h.get(h2h_key, 0.5)

        feature_vector = [
            h["win_rate"], h["draw_rate"], h["avg_scored"], h["avg_conceded"],
            h["avg_ht"], h["goal_diff"],
            a["win_rate"], a["draw_rate"], a["avg_scored"], a["avg_conceded"],
            a["avg_ht"], a["goal_diff"],
            h["win_rate"] - a["win_rate"],
            h["avg_scored"] - a["avg_scored"],
            h["avg_conceded"] - a["avg_conceded"],
            h["goal_diff"] - a["goal_diff"],
            h2h_score,
        ]

        X = np.array([feature_vector])

        # Escalar si es necesario
        if self.scaler:
            X = self.scaler.transform(X)

        prediction = int(self.model.predict(X)[0])
        probabilities = self.model.predict_proba(X)[0].tolist()

        return {
            "home_team": home_resolved,
            "away_team": away_resolved,
            "prediction": prediction,
            "label": LABELS[prediction],
            "label_es": LABELS_ES[prediction],
            "probabilities": {
                LABELS[i]: round(p, 4) for i, p in enumerate(probabilities)
            },
            "h2h_score": round(h2h_score, 4),
            "model": self.metadata.get("model_name", "unknown"),
        }


# --- Funciones de compatibilidad para Lambda ---
_predictor = None


def load_artifacts(model_path="/tmp/model.pkl", meta_path="/tmp/metadata.json",
                   stats_path="/tmp/team_stats.json"):
    """Carga artefactos para Lambda (compatibilidad)."""
    global _predictor
    _predictor = Predictor.__new__(Predictor)
    _predictor.model = joblib.load(model_path)
    with open(meta_path, "r") as f:
        _predictor.metadata = json.load(f)
    with open(stats_path, "r") as f:
        all_stats = json.load(f)
    _predictor.team_stats = all_stats["teams"]
    _predictor.h2h = all_stats.get("h2h", {})
    _predictor.features = _predictor.metadata["features"]
    _predictor.teams = sorted(_predictor.team_stats.keys())
    _predictor.scaler = None
    if _predictor.metadata.get("requires_scaling", False):
        scaler_path = model_path.replace("model.pkl", "scaler.pkl")
        if os.path.exists(scaler_path):
            _predictor.scaler = joblib.load(scaler_path)
    return _predictor


def predict(home_team: str, away_team: str, **kwargs) -> dict:
    """Wrapper funcional para Lambda."""
    global _predictor
    if _predictor is None:
        _predictor = Predictor("models")
    return _predictor.predict(home_team, away_team)
