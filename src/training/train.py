"""
train.py — Entrenamiento completo del modelo de predicción Liga MX.

Entrena Random Forest (principal) y Logistic Regression (baseline).
Selecciona el mejor por cross-validation y serializa:
  - model.pkl          → modelo entrenado
  - metadata.json      → features, métricas, equipos, hiperparámetros
  - team_stats.json    → stats más recientes por equipo (para inferencia)

Uso:
  python src/training/train.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
)
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

FEATURE_COLS = [
    "h_win_rate", "h_draw_rate", "h_avg_scored", "h_avg_conceded",
    "h_avg_ht", "h_goal_diff",
    "a_win_rate", "a_draw_rate", "a_avg_scored", "a_avg_conceded",
    "a_avg_ht", "a_goal_diff",
    "win_rate_diff", "goals_scored_diff", "defense_diff",
    "goal_diff_diff", "h2h_home_advantage",
]

TARGET_COL = "target"
TARGET_NAMES = ["Home Win", "Draw", "Away Win"]
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data() -> pd.DataFrame:
    """Carga el dataset procesado."""
    path = os.path.join(PROCESSED_DIR, "matches.csv")
    df = pd.read_csv(path)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas faltantes: {missing}. Corre preprocess.py primero.")

    return df


def train_logistic_regression(X_train, y_train, X_test, y_test) -> dict:
    """Entrena Logistic Regression con scaling."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
        C=1.0,
        solver="lbfgs",
    )
    lr.fit(X_train_scaled, y_train)

    test_acc = accuracy_score(y_test, lr.predict(X_test_scaled))
    cv_scores = cross_val_score(lr, X_train_scaled, y_train, cv=5, scoring="accuracy")

    return {
        "model": lr,
        "scaler": scaler,
        "test_accuracy": test_acc,
        "cv_accuracy": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "name": "logistic_regression",
    }


def train_random_forest(X_train, y_train, X_test, y_test) -> dict:
    """Entrena Random Forest con búsqueda de hiperparámetros."""
    # Grid search ligero para encontrar buenos hiperparámetros
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 15, 20],
        "min_samples_split": [3, 5],
        "min_samples_leaf": [2, 4],
    }

    rf_base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    grid = GridSearchCV(
        rf_base, param_grid, cv=5, scoring="accuracy",
        n_jobs=-1, verbose=0,
    )
    grid.fit(X_train, y_train)

    best_rf = grid.best_estimator_
    test_acc = accuracy_score(y_test, best_rf.predict(X_test))
    cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5, scoring="accuracy")

    return {
        "model": best_rf,
        "scaler": None,
        "test_accuracy": test_acc,
        "cv_accuracy": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "name": "random_forest",
        "best_params": grid.best_params_,
    }


def compute_latest_team_stats(df: pd.DataFrame) -> dict:
    """
    Extrae las stats más recientes de cada equipo para usarlas
    en inferencia (predecir con solo nombres de equipos).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.sort_values("date")

    teams = sorted(set(df["home_team"].unique()) | set(df["away_team"].unique()))
    latest = {}

    for team in teams:
        # Buscar la última aparición como local
        home_matches = df[df["home_team"] == team]
        # Buscar la última aparición como visitante
        away_matches = df[df["away_team"] == team]

        stats = {"win_rate": 0.0, "draw_rate": 0.0, "avg_scored": 0.0,
                 "avg_conceded": 0.0, "avg_ht": 0.0, "goal_diff": 0.0}

        if len(home_matches) > 0:
            last_home = home_matches.iloc[-1]
            stats["win_rate"] = float(last_home.get("h_win_rate", 0))
            stats["draw_rate"] = float(last_home.get("h_draw_rate", 0))
            stats["avg_scored"] = float(last_home.get("h_avg_scored", 0))
            stats["avg_conceded"] = float(last_home.get("h_avg_conceded", 0))
            stats["avg_ht"] = float(last_home.get("h_avg_ht", 0))
            stats["goal_diff"] = float(last_home.get("h_goal_diff", 0))
        elif len(away_matches) > 0:
            last_away = away_matches.iloc[-1]
            stats["win_rate"] = float(last_away.get("a_win_rate", 0))
            stats["draw_rate"] = float(last_away.get("a_draw_rate", 0))
            stats["avg_scored"] = float(last_away.get("a_avg_scored", 0))
            stats["avg_conceded"] = float(last_away.get("a_avg_conceded", 0))
            stats["avg_ht"] = float(last_away.get("a_avg_ht", 0))
            stats["goal_diff"] = float(last_away.get("a_goal_diff", 0))

        latest[team] = stats

    # Calcular H2H para todos los pares
    h2h = {}
    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        target = row["target"]
        key = f"{home}_vs_{away}"
        if key not in h2h:
            h2h[key] = []
        h2h[key].append(1.0 if target == 0 else (0.5 if target == 1 else 0.0))

    h2h_averages = {k: round(np.mean(v[-6:]), 4) for k, v in h2h.items()}

    return {"teams": latest, "h2h": h2h_averages}


def print_feature_importance(model, feature_names: list):
    """Muestra la importancia de cada feature (solo para Random Forest)."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("\n📊 Importancia de features:")
        for i, idx in enumerate(indices):
            bar = "█" * int(importances[idx] * 50)
            print(f"   {i+1:2d}. {feature_names[idx]:<25s} {importances[idx]:.4f} {bar}")


def main():
    print("=" * 60)
    print("  🧠 Entrenamiento — Predictor Liga MX")
    print("=" * 60)

    # Cargar datos
    print("\n📂 Cargando dataset procesado...")
    df = load_data()

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    print(f"   Samples: {len(X)}")
    print(f"   Features: {len(FEATURE_COLS)}")
    print(f"   Clases: {dict(zip(TARGET_NAMES, np.bincount(y)))}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

    # --- Entrenar modelos ---
    print("\n" + "-" * 60)
    print("📈 Entrenando Logistic Regression (baseline)...")
    lr_result = train_logistic_regression(X_train, y_train, X_test, y_test)
    print(f"   Test Accuracy:  {lr_result['test_accuracy']:.4f}")
    print(f"   CV Accuracy:    {lr_result['cv_accuracy']:.4f} (±{lr_result['cv_std']:.4f})")

    print("\n🌲 Entrenando Random Forest (GridSearchCV)...")
    rf_result = train_random_forest(X_train, y_train, X_test, y_test)
    print(f"   Best params:    {rf_result['best_params']}")
    print(f"   Test Accuracy:  {rf_result['test_accuracy']:.4f}")
    print(f"   CV Accuracy:    {rf_result['cv_accuracy']:.4f} (±{rf_result['cv_std']:.4f})")

    # --- Seleccionar mejor modelo ---
    print("\n" + "-" * 60)
    results = {"logistic_regression": lr_result, "random_forest": rf_result}
    best_name = max(results, key=lambda k: results[k]["cv_accuracy"])
    best = results[best_name]
    print(f"🏆 Mejor modelo: {best_name} (CV: {best['cv_accuracy']:.4f})")

    # --- Classification Report del mejor modelo ---
    model = best["model"]
    scaler = best.get("scaler")

    X_test_eval = scaler.transform(X_test) if scaler else X_test
    y_pred = model.predict(X_test_eval)

    print(f"\n📝 Classification Report ({best_name}):\n")
    print(classification_report(y_test, y_pred, target_names=TARGET_NAMES))

    print("📋 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"   {'':>12s}  {'Pred HW':>8s}  {'Pred D':>8s}  {'Pred AW':>8s}")
    for i, label in enumerate(TARGET_NAMES):
        print(f"   {label:>12s}  {cm[i][0]:>8d}  {cm[i][1]:>8d}  {cm[i][2]:>8d}")

    # Feature importance
    if best_name == "random_forest":
        print_feature_importance(model, FEATURE_COLS)

    # --- Guardar artefactos ---
    print("\n" + "-" * 60)
    print("💾 Guardando artefactos...")
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Modelo
    model_path = os.path.join(MODELS_DIR, "model.pkl")
    joblib.dump(model, model_path)

    # Scaler (si aplica)
    if scaler:
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        joblib.dump(scaler, scaler_path)

    # Metadata
    metadata = {
        "model_name": best_name,
        "features": FEATURE_COLS,
        "target_names": TARGET_NAMES,
        "requires_scaling": scaler is not None,
        "metrics": {
            name: {
                "test_accuracy": round(r["test_accuracy"], 4),
                "cv_accuracy": round(r["cv_accuracy"], 4),
                "cv_std": round(r["cv_std"], 4),
            }
            for name, r in results.items()
        },
        "hyperparameters": rf_result.get("best_params", {}),
        "dataset": {
            "total_samples": len(X),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "n_features": len(FEATURE_COLS),
        },
    }
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Team stats para inferencia
    print("📊 Calculando stats de equipos para inferencia...")
    all_stats = compute_latest_team_stats(df)

    metadata["teams"] = sorted(all_stats["teams"].keys())

    # Re-guardar metadata con la lista de equipos
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    with open(os.path.join(MODELS_DIR, "team_stats.json"), "w") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    print(f"\n   📁 models/model.pkl")
    if scaler:
        print(f"   📁 models/scaler.pkl")
    print(f"   📁 models/metadata.json")
    print(f"   📁 models/team_stats.json ({len(all_stats['teams'])} equipos, {len(all_stats['h2h'])} pares H2H)")

    print("\n" + "=" * 60)
    print("  ✅ Entrenamiento completado exitosamente")
    print("=" * 60)


if __name__ == "__main__":
    main()
