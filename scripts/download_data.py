#!/usr/bin/env python3
"""
download_data.py — Descarga el dataset de Liga MX desde Kaggle.

Requiere: pip install kaggle + ~/.kaggle/kaggle.json configurado.

Alternativamente, descarga manualmente desde:
  https://www.kaggle.com/datasets/gerardojaimeescareo/ligamx-matches-2016-2022
"""

import os
import subprocess
import sys

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
DATASET = "gerardojaimeescareo/ligamx-matches-2016-2022"


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    existing = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
    if existing:
        print(f"⏭️  Ya existen {len(existing)} CSV(s) en {RAW_DIR}")
        return

    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("⚠️  Kaggle CLI no disponible.")
        print(f"   Descarga manual: https://www.kaggle.com/datasets/{DATASET}")
        print(f"   Coloca el CSV en: {os.path.abspath(RAW_DIR)}")
        sys.exit(1)

    print(f"📥 Descargando {DATASET}...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", RAW_DIR, "--unzip"],
        check=True,
    )
    print(f"✅ Dataset descargado en {RAW_DIR}")


if __name__ == "__main__":
    main()
