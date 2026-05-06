#!/usr/bin/env python3
"""
predict_cli.py — CLI para predecir resultados de la Liga MX.

Uso:
  python scripts/predict_cli.py "América" "Chivas"
  python scripts/predict_cli.py --interactive
  python scripts/predict_cli.py --teams
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "inference"))
from predict import Predictor


def print_prediction(result: dict):
    """Imprime la predicción de forma visual."""
    home = result["home_team"]
    away = result["away_team"]
    probs = result["probabilities"]

    print(f"\n{'=' * 50}")
    print(f"  ⚽  {home}  vs  {away}")
    print(f"{'=' * 50}")
    print(f"\n  🏆  Predicción: {result['label_es']}")
    print(f"      ({result['label']})\n")

    # Barras visuales de probabilidad
    for label, prob in probs.items():
        bar_len = int(prob * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {label:<10s}  {bar}  {prob:.1%}")

    print(f"\n  📊  H2H score: {result['h2h_score']:.2f}")
    print(f"  🤖  Modelo: {result['model']}")
    print(f"{'=' * 50}\n")


def interactive_mode(predictor: Predictor):
    """Modo interactivo: permite hacer múltiples predicciones."""
    print("\n🏟️  Predictor Liga MX — Modo Interactivo")
    print("   Escribe 'salir' para terminar.\n")

    teams = predictor.get_teams()
    print("   Equipos disponibles:")
    for i, team in enumerate(teams, 1):
        print(f"     {i:2d}. {team}")
    print()

    while True:
        try:
            home = input("   ⚽ Equipo LOCAL:     ").strip()
            if home.lower() in ("salir", "exit", "q"):
                break

            away = input("   ⚽ Equipo VISITANTE: ").strip()
            if away.lower() in ("salir", "exit", "q"):
                break

            result = predictor.predict(home, away)
            print_prediction(result)

        except ValueError as e:
            print(f"\n   ❌ {e}\n")
        except KeyboardInterrupt:
            print("\n\n   👋 ¡Hasta luego!")
            break


def main():
    parser = argparse.ArgumentParser(description="Predictor de resultados Liga MX")
    parser.add_argument("home_team", nargs="?", help="Equipo local")
    parser.add_argument("away_team", nargs="?", help="Equipo visitante")
    parser.add_argument("--interactive", "-i", action="store_true", help="Modo interactivo")
    parser.add_argument("--teams", "-t", action="store_true", help="Listar equipos disponibles")
    parser.add_argument("--models-dir", default="models", help="Directorio de artefactos")

    args = parser.parse_args()

    try:
        predictor = Predictor(args.models_dir)
    except FileNotFoundError:
        print("❌ No se encontraron artefactos del modelo.")
        print("   Ejecuta primero: make pipeline")
        sys.exit(1)

    if args.teams:
        print("\n📋 Equipos disponibles:\n")
        for team in predictor.get_teams():
            print(f"   • {team}")
        print()
        return

    if args.interactive:
        interactive_mode(predictor)
        return

    if not args.home_team or not args.away_team:
        parser.print_help()
        print("\nEjemplos:")
        print('  python scripts/predict_cli.py "América" "Chivas"')
        print('  python scripts/predict_cli.py --interactive')
        print('  python scripts/predict_cli.py --teams')
        sys.exit(1)

    try:
        result = predictor.predict(args.home_team, args.away_team)
        print_prediction(result)
    except ValueError as e:
        print(f"\n❌ {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
