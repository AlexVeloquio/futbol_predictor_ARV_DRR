"""Tests para el módulo de preprocesamiento — Liga MX."""

import pandas as pd
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "training"))
from preprocess import clean, engineer_features, TEAM_NAME_MAP


@pytest.fixture
def sample_df():
    """DataFrame con formato idéntico al CSV de Kaggle."""
    return pd.DataFrame({
        "id": range(1, 11),
        "date": [
            "2023-01-07T20:00:00+00:00", "2023-01-08T00:00:00+00:00",
            "2023-01-14T20:00:00+00:00", "2023-01-15T00:00:00+00:00",
            "2023-01-21T20:00:00+00:00", "2023-01-22T00:00:00+00:00",
            "2023-01-28T20:00:00+00:00", "2023-01-29T00:00:00+00:00",
            "2023-02-04T20:00:00+00:00", "2023-02-05T00:00:00+00:00",
        ],
        "home_team": [
            "Club America", "Cruz Azul", "Guadalajara Chivas",
            "Club America", "Cruz Azul", "Guadalajara Chivas",
            "Club America", "Cruz Azul", "Guadalajara Chivas",
            "Club America",
        ],
        "away_team": [
            "Cruz Azul", "Guadalajara Chivas", "Club America",
            "Guadalajara Chivas", "Club America", "Cruz Azul",
            "Cruz Azul", "Guadalajara Chivas", "Club America",
            "Cruz Azul",
        ],
        "home_goals": [2, 1, 0, 3, 1, 1, 2, 0, 1, 1],
        "away_goals": [1, 1, 2, 0, 1, 2, 0, 3, 1, 0],
        "home_win": [True, False, False, True, False, False, True, False, False, True],
        "away_win": [False, False, True, False, False, True, False, True, False, False],
        "home_goals_half_time": [1, 0, 0, 2, 1, 0, 1, 0, 0, 1],
        "away_goals_half_time": [0, 1, 1, 0, 0, 1, 0, 2, 1, 0],
        "home_goals_fulltime": [2, 1, 0, 3, 1, 1, 2, 0, 1, 1],
        "away_goals_fulltime": [1, 1, 2, 0, 1, 2, 0, 3, 1, 0],
        "season": [2023] * 10,
        "round": ["Clausura - 1"] * 10,
        "home_goals_extra_time": [np.nan] * 10,
        "away_goals_extratime": [np.nan] * 10,
        "home_goals_penalty": [np.nan] * 10,
        "away_goals_penalty": [np.nan] * 10,
        "referee": ["Test Ref"] * 10,
        "timezone": ["UTC"] * 10,
        "venue_id": [1] * 10,
        "venue_name": ["Estadio Test"] * 10,
        "venue_city": ["CDMX"] * 10,
    })


class TestClean:
    def test_generates_target(self, sample_df):
        result = clean(sample_df)
        assert "target" in result.columns

    def test_home_win_is_zero(self, sample_df):
        result = clean(sample_df)
        home_wins = result[result["home_goals"] > result["away_goals"]]
        assert (home_wins["target"] == 0).all()

    def test_draw_is_one(self, sample_df):
        result = clean(sample_df)
        draws = result[result["home_goals"] == result["away_goals"]]
        assert (draws["target"] == 1).all()

    def test_away_win_is_two(self, sample_df):
        result = clean(sample_df)
        away_wins = result[result["home_goals"] < result["away_goals"]]
        assert (away_wins["target"] == 2).all()

    def test_drops_rows_without_goals(self, sample_df):
        sample_df.loc[0, "home_goals"] = np.nan
        sample_df.loc[0, "away_goals"] = np.nan
        result = clean(sample_df)
        assert len(result) == 9

    def test_normalizes_team_names(self, sample_df):
        sample_df.loc[0, "home_team"] = "León"
        result = clean(sample_df)
        assert "León" not in result["home_team"].values
        assert "Leon" in result["home_team"].values

    def test_sorted_by_date(self, sample_df):
        result = clean(sample_df)
        dates = result["date"].tolist()
        assert dates == sorted(dates)


class TestEngineerFeatures:
    def test_generates_all_feature_columns(self, sample_df):
        df = clean(sample_df)
        result = engineer_features(df)
        expected = [
            "h_win_rate", "h_draw_rate", "h_avg_scored", "h_avg_conceded",
            "h_avg_ht", "h_goal_diff",
            "a_win_rate", "a_draw_rate", "a_avg_scored", "a_avg_conceded",
            "a_avg_ht", "a_goal_diff",
            "win_rate_diff", "goals_scored_diff", "defense_diff",
            "goal_diff_diff", "h2h_home_advantage",
        ]
        for col in expected:
            assert col in result.columns, f"Falta: {col}"

    def test_filters_matches_without_history(self, sample_df):
        df = clean(sample_df)
        result = engineer_features(df)
        assert len(result) <= len(df)

    def test_win_rate_between_0_and_1(self, sample_df):
        df = clean(sample_df)
        result = engineer_features(df)
        assert result["h_win_rate"].between(0, 1).all()
        assert result["a_win_rate"].between(0, 1).all()

    def test_diff_is_consistent(self, sample_df):
        df = clean(sample_df)
        result = engineer_features(df)
        for _, row in result.iterrows():
            diff = row["h_win_rate"] - row["a_win_rate"]
            assert abs(row["win_rate_diff"] - diff) < 1e-6
