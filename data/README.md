# Datos — Predictor de Resultados Liga MX

## Fuente

[Liga MX Matches 2016-2024 (Kaggle)](https://www.kaggle.com/datasets/gerardojaimeescareo/ligamx-matches-2016-2022)  
Autor: Gerardo Jaime Escareo

## Descarga

```bash
# Opción 1: Kaggle CLI
pip install kaggle
kaggle datasets download -d gerardojaimeescareo/ligamx-matches-2016-2022 -p data/raw/ --unzip

# Opción 2: Manual desde Kaggle, colocar CSV en data/raw/
```

## Diccionario de datos

| Columna              | Descripción                       | Tipo   |
|----------------------|-----------------------------------|--------|
| id                   | ID del partido                    | int    |
| date                 | Fecha (ISO 8601 UTC)              | string |
| home_team            | Equipo local                      | string |
| away_team            | Equipo visitante                  | string |
| home_win             | True si ganó local                | bool   |
| away_win             | True si ganó visitante            | bool   |
| home_goals           | Goles del local                   | float  |
| away_goals           | Goles del visitante               | float  |
| home_goals_half_time | Goles local medio tiempo          | float  |
| away_goals_half_time | Goles visitante medio tiempo      | float  |
| season               | Año de la temporada               | int    |
| round                | Ronda (Apertura/Clausura + jornada)| string |

## Target

| Valor | Etiqueta  | Derivado de                     |
|-------|-----------|----------------------------------|
| 0     | Home Win  | home_goals > away_goals          |
| 1     | Draw      | home_goals == away_goals         |
| 2     | Away Win  | home_goals < away_goals          |

## Estadísticas del dataset

- **2,813** partidos válidos (63 eliminados por COVID)
- **22** equipos
- **9** temporadas (2016-2024)
- **Distribución:** Home Win 44.8% / Draw 27.0% / Away Win 28.3%
