"""Tests para el módulo de predicción — Liga MX."""

import numpy as np
import pytest
import json
import os
import sys
import joblib
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "inference"))
from predict import Predictor, resolve_team_name, LABELS


@pytest.fixture
def mock_artifacts(tmp_path):
    """Crea artefactos mock en un directorio temporal."""
    # Entrenar un modelo real mínimo
    X_dummy = np.random.rand(30, 17)
    y_dummy = np.array([0, 1, 2] * 10)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_dummy, y_dummy)
    joblib.dump(model, tmp_path / "model.pkl")

    # Metadata
    metadata = {
        "model_name": "random_forest",
        "features": [
            "h_win_rate", "h_draw_rate", "h_avg_scored", "h_avg_conceded",
            "h_avg_ht", "h_goal_diff",
            "a_win_rate", "a_draw_rate", "a_avg_scored", "a_avg_conceded",
            "a_avg_ht", "a_goal_diff",
            "win_rate_diff", "goals_scored_diff", "defense_diff",
            "goal_diff_diff", "h2h_home_advantage",
        ],
        "target_names": ["Home Win", "Draw", "Away Win"],
        "requires_scaling": False,
        "teams": ["Club America", "Guadalajara Chivas", "Cruz Azul"],
    }
    with open(tmp_path / "metadata.json", "w") as f:
        json.dump(metadata, f)

    # Team stats
    team_stats = {
        "teams": {
            "Club America": {"win_rate": 0.7, "draw_rate": 0.1, "avg_scored": 1.8,
                             "avg_conceded": 0.6, "avg_ht": 0.8, "goal_diff": 1.2},
            "Guadalajara Chivas": {"win_rate": 0.5, "draw_rate": 0.2, "avg_scored": 1.2,
                                   "avg_conceded": 1.0, "avg_ht": 0.5, "goal_diff": 0.2},
            "Cruz Azul": {"win_rate": 0.6, "draw_rate": 0.15, "avg_scored": 1.5,
                          "avg_conceded": 0.8, "avg_ht": 0.7, "goal_diff": 0.7},
        },
        "h2h": {
            "Club America_vs_Guadalajara Chivas": 0.75,
            "Guadalajara Chivas_vs_Club America": 0.25,
            "Cruz Azul_vs_Club America": 0.5,
        },
    }
    with open(tmp_path / "team_stats.json", "w") as f:
        json.dump(team_stats, f)

    return str(tmp_path)


@pytest.fixture
def predictor(mock_artifacts):
    return Predictor(mock_artifacts)


class TestResolveTeamName:
    def test_exact_match(self):
        teams = ["Club America", "Guadalajara Chivas"]
        assert resolve_team_name("Club America", teams) == "Club America"

    def test_alias_america(self):
        teams = ["Club America", "Guadalajara Chivas"]
        assert resolve_team_name("América", teams) == "Club America"

    def test_alias_chivas(self):
        teams = ["Club America", "Guadalajara Chivas"]
        assert resolve_team_name("Chivas", teams) == "Guadalajara Chivas"

    def test_alias_pumas(self):
        teams = ["U.N.A.M. - Pumas"]
        assert resolve_team_name("Pumas", teams) == "U.N.A.M. - Pumas"

    def test_unknown_returns_none(self):
        teams = ["Club America"]
        assert resolve_team_name("FC Barcelona", teams) is None

    def test_case_insensitive(self):
        teams = ["Club America"]
        assert resolve_team_name("club america", teams) == "Club America"


class TestPredictor:
    def test_returns_correct_structure(self, predictor):
        result = predictor.predict("Club America", "Guadalajara Chivas")
        assert "home_team" in result
        assert "away_team" in result
        assert "prediction" in result
        assert "label" in result
        assert "label_es" in result
        assert "probabilities" in result
        assert "h2h_score" in result
        assert "model" in result

    def test_team_names_resolved(self, predictor):
        result = predictor.predict("Club America", "Guadalajara Chivas")
        assert result["home_team"] == "Club America"
        assert result["away_team"] == "Guadalajara Chivas"

    def test_alias_resolution_in_predict(self, predictor):
        result = predictor.predict("América", "Chivas")
        assert result["home_team"] == "Club America"
        assert result["away_team"] == "Guadalajara Chivas"

    def test_probabilities_sum_to_one(self, predictor):
        result = predictor.predict("Club America", "Cruz Azul")
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 0.01

    def test_unknown_team_raises(self, predictor):
        with pytest.raises(ValueError, match="no encontrado"):
            predictor.predict("FC Barcelona", "Club America")

    def test_same_team_raises(self, predictor):
        with pytest.raises(ValueError, match="no pueden ser el mismo"):
            predictor.predict("Club America", "Club America")

    def test_h2h_score_present(self, predictor):
        result = predictor.predict("Club America", "Guadalajara Chivas")
        assert result["h2h_score"] == 0.75

    def test_h2h_default_when_missing(self, predictor):
        result = predictor.predict("Club America", "Cruz Azul")
        # No hay h2h directo para America vs Cruz Azul, default 0.5
        assert result["h2h_score"] == 0.5

    def test_get_teams(self, predictor):
        teams = predictor.get_teams()
        assert len(teams) == 3
        assert "Club America" in teams
