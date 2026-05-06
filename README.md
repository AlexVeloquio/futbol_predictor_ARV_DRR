# ⚽ Predictor de Resultados — Liga MX

> Modelo de ML para predecir resultados de la Liga MX (Victoria Local, Empate, Victoria Visitante).  
> Seleccionas dos equipos y el modelo te dice quién tiene más probabilidad de ganar.

**Materia:** Infraestructura y Desarrollo Continuo — ITESO  
**Equipo:** Alejandro Rodríguez Veloquio · Diego Rosales Rojas

---

## Descripción

Sistema de Machine Learning que utiliza datos históricos de la Liga MX (2016-2024) para predecir el resultado de un partido entre dos equipos. Genera features estadísticas rolling por equipo (tasa de victorias, promedios de goles, head-to-head) y entrena un Random Forest Classifier como modelo principal con Logistic Regression como baseline.

## Dataset

**Fuente:** [Liga MX Matches 2016-2024 (Kaggle)](https://www.kaggle.com/datasets/gerardojaimeescareo/ligamx-matches-2016-2022)  
- 2,813 partidos válidos | 22 equipos | 9 temporadas  
- Ver [data/README.md](data/README.md) para el diccionario de datos completo

## Inicio Rápido

```bash
# 1. Clonar
git clone https://github.com/<tu-usuario>/futbol-predictor.git
cd futbol-predictor

# 2. Entorno
make setup
source venv/bin/activate

# 3. Dataset: descargar de Kaggle y colocar en data/raw/

# 4. Pipeline completo (preprocesamiento + entrenamiento)
make pipeline

# 5. Predecir
make predict HOME="América" AWAY="Chivas"
make predict HOME="Cruz Azul" AWAY="Pumas"
make predict HOME="Tigres" AWAY="Monterrey"

# 6. Modo interactivo
make interactive

# 7. Tests
make test
```

## Ejemplo de predicción

```
==================================================
  ⚽  Club America  vs  Guadalajara Chivas
==================================================

  🏆  Predicción: Victoria Local
      (Home Win)

  Home Win    ███████████████░░░░░░░░░░░░░░░  53.0%
  Draw        █████████░░░░░░░░░░░░░░░░░░░░░  30.6%
  Away Win    ████░░░░░░░░░░░░░░░░░░░░░░░░░░  16.4%

  📊  H2H score: 0.75
  🤖  Modelo: random_forest
==================================================
```

## Arquitectura

```
┌──────────────── Local ─────────────────┐     ┌──────── AWS ────────┐
│                                         │     │                     │
│  CSV ──▶ preprocess.py ──▶ train.py     │────▶│  S3 → Lambda → API │
│              │                  │        │     │         ↕           │
│         matches.csv      model.pkl      │     │     DynamoDB        │
│                          team_stats     │     │                     │
└─────────────────────────────────────────┘     └─────────────────────┘
```

## Modelos

| Modelo                   | Rol       | CV Accuracy |
|--------------------------|-----------|-------------|
| Random Forest Classifier | Principal | ~44.5%      |
| Logistic Regression      | Baseline  | ~44.5%      |

> La precisión de ~44.5% es razonable para predicción de fútbol (3 clases con desbalance).  
> El enfoque del proyecto es la infraestructura, no optimizar el modelo.

## Features (17)

El modelo usa estadísticas rolling (últimos 5 partidos) de cada equipo:

- **Por equipo:** win_rate, draw_rate, avg_scored, avg_conceded, avg_ht, goal_diff
- **Diferencias:** win_rate_diff, goals_scored_diff, defense_diff, goal_diff_diff
- **Head-to-head:** historial directo entre ambos equipos

## Estructura

```
├── src/training/          # Preprocesamiento + entrenamiento
│   ├── preprocess.py      # Limpieza, feature engineering
│   └── train.py           # Random Forest + Logistic Regression
├── src/inference/         # Servicio de predicción
│   ├── predict.py         # Predictor con aliases de equipos
│   ├── handler.py         # Lambda handler
│   └── Dockerfile
├── scripts/
│   ├── predict_cli.py     # CLI interactivo
│   └── download_data.py   # Descarga de Kaggle
├── tests/                 # 26 tests unitarios
├── models/                # Artefactos (model.pkl, team_stats.json)
├── infrastructure/        # Terraform (IaC)
├── .github/workflows/     # CI/CD (GitHub Actions)
├── Makefile               # Automatización
└── docker-compose.yml     # Entorno local
```

## Tech Stack

| Capa           | Tecnología                    |
|----------------|-------------------------------|
| ML             | scikit-learn, pandas, numpy   |
| Contenedores   | Docker, Amazon ECR            |
| Cómputo        | AWS Lambda (container image)  |
| Almacenamiento | Amazon S3                     |
| Base de datos  | Amazon DynamoDB               |
| API            | Amazon API Gateway            |
| IaC            | Terraform                     |
| CI/CD          | GitHub Actions                |

## Licencia

MIT
