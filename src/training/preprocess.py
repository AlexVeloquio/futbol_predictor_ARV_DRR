"""
preprocess.py — Limpieza y feature engineering del dataset Liga MX (Kaggle).

Entrada:  data/raw/*.csv  (dataset de Kaggle)
Salida:   data/processed/matches.csv  (listo para entrenamiento)

Features generadas (rolling window de últimos N partidos por equipo):
  - Tasa de victorias, empates
  - Promedio de goles anotados / recibidos
  - Promedio de goles al medio tiempo
  - Head-to-head histórico entre los dos equipos
  - Diferencias (local - visitante) de todas las métricas
"""

import os
import pandas as pd
import numpy as np

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

# Normalización de nombres de equipos (typos en el dataset)
TEAM_NAME_MAP = {
    "León": "Leon",
    "Atlético San Luis": "Atletico San Luis",
}

WINDOW = 5  # Partidos históricos para rolling stats


def load_raw(raw_dir: str) -> pd.DataFrame:
    """Carga todos los CSVs de data/raw/."""
    csv_files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron CSVs en {raw_dir}")

    frames = []
    for f in sorted(csv_files):
        path = os.path.join(raw_dir, f)
        df = pd.read_csv(path)
        print(f"  📄 {f}: {len(df)} registros")
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza inicial del dataset."""
    df = df.copy()

    # Normalizar nombres de equipos
    df["home_team"] = df["home_team"].replace(TEAM_NAME_MAP)
    df["away_team"] = df["away_team"].replace(TEAM_NAME_MAP)

    # Parsear fechas
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)

    # Eliminar partidos sin goles (COVID / no jugados)
    before = len(df)
    df = df.dropna(subset=["home_goals", "away_goals"])
    print(f"  🗑️  {before - len(df)} partidos sin datos eliminados (COVID / no jugados)")

    # Convertir goles a int
    for col in ["home_goals", "away_goals", "home_goals_half_time", "away_goals_half_time"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Generar target: 0=Home Win, 1=Draw, 2=Away Win
    # En el dataset: home_win=NaN y away_win=NaN → empate
    df["target"] = np.where(
        df["home_goals"] > df["away_goals"], 0,
        np.where(df["home_goals"] < df["away_goals"], 2, 1)
    )

    # Ordenar cronológicamente
    df = df.sort_values("date").reset_index(drop=True)

    return df


def build_team_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye historial unificado por equipo (local y visitante).
    Cada fila = un equipo en un partido.
    """
    home_records = pd.DataFrame({
        "date": df["date"],
        "team": df["home_team"],
        "opponent": df["away_team"],
        "goals_scored": df["home_goals"],
        "goals_conceded": df["away_goals"],
        "goals_ht": df["home_goals_half_time"],
        "is_home": 1,
        "won": (df["target"] == 0).astype(int),
        "drew": (df["target"] == 1).astype(int),
        "lost": (df["target"] == 2).astype(int),
    })

    away_records = pd.DataFrame({
        "date": df["date"],
        "team": df["away_team"],
        "opponent": df["home_team"],
        "goals_scored": df["away_goals"],
        "goals_conceded": df["home_goals"],
        "goals_ht": df["away_goals_half_time"],
        "is_home": 0,
        "won": (df["target"] == 2).astype(int),
        "drew": (df["target"] == 1).astype(int),
        "lost": (df["target"] == 0).astype(int),
    })

    history = pd.concat([home_records, away_records], ignore_index=True)
    history = history.sort_values(["team", "date"]).reset_index(drop=True)
    return history


def compute_rolling_stats(history: pd.DataFrame, window: int) -> dict:
    """
    Calcula stats rolling por equipo.
    Retorna dict: team -> DataFrame con stats acumuladas.
    """
    stats_by_team = {}

    for team, group in history.groupby("team"):
        g = group.copy()
        g["win_rate"] = g["won"].rolling(window, min_periods=1).mean()
        g["draw_rate"] = g["drew"].rolling(window, min_periods=1).mean()
        g["avg_scored"] = g["goals_scored"].rolling(window, min_periods=1).mean()
        g["avg_conceded"] = g["goals_conceded"].rolling(window, min_periods=1).mean()
        g["avg_ht_scored"] = g["goals_ht"].rolling(window, min_periods=1).mean()
        g["goal_diff"] = g["avg_scored"] - g["avg_conceded"]
        stats_by_team[team] = g.reset_index(drop=True)

    return stats_by_team


def compute_h2h(df: pd.DataFrame, window: int = 6) -> dict:
    """
    Calcula head-to-head entre pares de equipos.
    Retorna dict: (teamA, teamB) -> win_rate de teamA vs teamB.
    """
    h2h = {}

    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        target = row["target"]

        # Home perspective
        key_h = (home, away)
        if key_h not in h2h:
            h2h[key_h] = []
        h2h[key_h].append(1 if target == 0 else (0.5 if target == 1 else 0))

        # Away perspective
        key_a = (away, home)
        if key_a not in h2h:
            h2h[key_a] = []
        h2h[key_a].append(1 if target == 2 else (0.5 if target == 1 else 0))

    # Convertir a promedios rolling
    h2h_stats = {}
    for key, results in h2h.items():
        recent = results[-window:]
        h2h_stats[key] = np.mean(recent)

    return h2h_stats


def engineer_features(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """
    Genera features para cada partido usando SOLO información previa
    (evita data leakage).
    """
    df = df.copy()

    history = build_team_history(df)
    team_stats = compute_rolling_stats(history, window)

    # Contadores para trackear la posición de cada equipo en su historial
    team_idx = {team: 0 for team in team_stats}

    # Head-to-head progresivo
    h2h_running = {}

    feature_rows = []

    for i, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        hi, ai = team_idx.get(home, 0), team_idx.get(away, 0)

        # --- Stats del equipo LOCAL (antes del partido actual) ---
        if home in team_stats and hi > 0:
            hs = team_stats[home].iloc[hi - 1]
            h_win_rate = hs["win_rate"]
            h_draw_rate = hs["draw_rate"]
            h_avg_scored = hs["avg_scored"]
            h_avg_conceded = hs["avg_conceded"]
            h_avg_ht = hs["avg_ht_scored"]
            h_goal_diff = hs["goal_diff"]
        else:
            h_win_rate = h_draw_rate = h_avg_scored = 0.0
            h_avg_conceded = h_avg_ht = h_goal_diff = 0.0

        # --- Stats del equipo VISITANTE (antes del partido actual) ---
        if away in team_stats and ai > 0:
            as_ = team_stats[away].iloc[ai - 1]
            a_win_rate = as_["win_rate"]
            a_draw_rate = as_["draw_rate"]
            a_avg_scored = as_["avg_scored"]
            a_avg_conceded = as_["avg_conceded"]
            a_avg_ht = as_["avg_ht_scored"]
            a_goal_diff = as_["goal_diff"]
        else:
            a_win_rate = a_draw_rate = a_avg_scored = 0.0
            a_avg_conceded = a_avg_ht = a_goal_diff = 0.0

        # --- Head-to-head ---
        h2h_key = (home, away)
        h2h_score = 0.5  # default neutral
        if h2h_key in h2h_running and len(h2h_running[h2h_key]) > 0:
            h2h_score = np.mean(h2h_running[h2h_key][-window:])

        # --- Construir fila de features ---
        feature_rows.append({
            # Features del equipo local
            "h_win_rate": h_win_rate,
            "h_draw_rate": h_draw_rate,
            "h_avg_scored": h_avg_scored,
            "h_avg_conceded": h_avg_conceded,
            "h_avg_ht": h_avg_ht,
            "h_goal_diff": h_goal_diff,

            # Features del equipo visitante
            "a_win_rate": a_win_rate,
            "a_draw_rate": a_draw_rate,
            "a_avg_scored": a_avg_scored,
            "a_avg_conceded": a_avg_conceded,
            "a_avg_ht": a_avg_ht,
            "a_goal_diff": a_goal_diff,

            # Diferencias
            "win_rate_diff": h_win_rate - a_win_rate,
            "goals_scored_diff": h_avg_scored - a_avg_scored,
            "defense_diff": h_avg_conceded - a_avg_conceded,
            "goal_diff_diff": h_goal_diff - a_goal_diff,

            # Head-to-head
            "h2h_home_advantage": h2h_score,
        })

        # --- Actualizar contadores e historial h2h ---
        team_idx[home] = hi + 1
        team_idx[away] = ai + 1

        target = row["target"]
        if h2h_key not in h2h_running:
            h2h_running[h2h_key] = []
        h2h_running[h2h_key].append(1 if target == 0 else (0.5 if target == 1 else 0))

        rev_key = (away, home)
        if rev_key not in h2h_running:
            h2h_running[rev_key] = []
        h2h_running[rev_key].append(1 if target == 2 else (0.5 if target == 1 else 0))

    feat_df = pd.DataFrame(feature_rows)
    result = pd.concat([df.reset_index(drop=True), feat_df], axis=1)

    # Filtrar partidos sin historial (primeros partidos de cada equipo)
    before = len(result)
    result = result[(result["h_win_rate"] > 0) | (result["a_win_rate"] > 0)].copy()
    result.reset_index(drop=True, inplace=True)
    print(f"  🔍 {before - len(result)} partidos filtrados (sin historial previo)")

    return result


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("📂 Cargando dataset Liga MX...\n")
    df = load_raw(RAW_DIR)
    print(f"\n   Total bruto: {len(df)} registros")

    print("\n🧹 Limpiando datos...")
    df = clean(df)

    target_labels = {0: "Home Win", 1: "Draw", 2: "Away Win"}
    print(f"\n📊 Distribución del target:")
    for t, label in target_labels.items():
        count = (df["target"] == t).sum()
        print(f"   {label}: {count} ({count / len(df) * 100:.1f}%)")

    print(f"\n   Equipos: {df['home_team'].nunique()}")
    print(f"   Temporadas: {sorted(df['season'].unique())}")

    print(f"\n🔧 Feature engineering (window={WINDOW})...")
    df = engineer_features(df, window=WINDOW)

    # Features generadas
    feature_cols = [c for c in df.columns if c.startswith(("h_", "a_", "win_", "goals_", "defense_", "goal_diff", "h2h_"))]
    print(f"\n   Features ({len(feature_cols)}): {feature_cols}")

    out_path = os.path.join(PROCESSED_DIR, "matches.csv")
    df.to_csv(out_path, index=False)
    print(f"\n✅ Dataset procesado: {out_path} ({len(df)} registros)")


if __name__ == "__main__":
    main()
