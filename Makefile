# ============================================================
#  Predictor Liga MX — Makefile
# ============================================================

.PHONY: help setup data preprocess train predict pipeline \
        test lint format docker-build deploy clean

PYTHON      := python3
VENV        := venv
AWS_REGION  := us-east-1
ECR_REPO    := futbol-predictor
ACCOUNT_ID  := $(shell aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "000000000000")
ECR_URI     := $(ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com

help: ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ============================================================
#  Entorno
# ============================================================
setup: ## Crea virtualenv e instala dependencias
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r src/training/requirements.txt
	$(VENV)/bin/pip install -r src/inference/requirements.txt
	$(VENV)/bin/pip install -r requirements-dev.txt
	@echo "\n✅ Entorno listo. Activa con: source $(VENV)/bin/activate"

# ============================================================
#  Pipeline Local
# ============================================================
data: ## Descarga dataset de Kaggle
	$(PYTHON) scripts/download_data.py

preprocess: ## Preprocesa el dataset
	$(PYTHON) src/training/preprocess.py

train: ## Entrena el modelo
	$(PYTHON) src/training/train.py

predict: ## Predicción (uso: make predict HOME="América" AWAY="Chivas")
	$(PYTHON) scripts/predict_cli.py "$(HOME)" "$(AWAY)"

interactive: ## Modo interactivo de predicción
	$(PYTHON) scripts/predict_cli.py --interactive

teams: ## Lista equipos disponibles
	$(PYTHON) scripts/predict_cli.py --teams

pipeline: preprocess train ## Pipeline completo: preprocess + train

# ============================================================
#  Calidad
# ============================================================
lint: ## Linting con flake8
	$(VENV)/bin/flake8 src/ tests/ --max-line-length=120

format: ## Formatea con black e isort
	$(VENV)/bin/black src/ tests/
	$(VENV)/bin/isort src/ tests/

test: ## Tests con pytest
	$(VENV)/bin/pytest tests/ -v --tb=short

# ============================================================
#  Docker & Deploy (al final del proyecto)
# ============================================================
docker-build: ## Construye imagen Docker de inferencia
	docker build -t $(ECR_REPO)-inference:latest -f src/inference/Dockerfile .

deploy: docker-build ## Deploy: imagen a ECR + modelo a S3 + update Lambda
	aws ecr get-login-password --region $(AWS_REGION) | \
		docker login --username AWS --password-stdin $(ECR_URI)
	docker tag $(ECR_REPO)-inference:latest $(ECR_URI)/$(ECR_REPO)-inference:latest
	docker push $(ECR_URI)/$(ECR_REPO)-inference:latest
	aws s3 cp models/model.pkl s3://$(ECR_REPO)-models/models/model.pkl
	aws s3 cp models/metadata.json s3://$(ECR_REPO)-models/models/metadata.json
	aws s3 cp models/team_stats.json s3://$(ECR_REPO)-models/models/team_stats.json
	aws lambda update-function-code \
		--function-name $(ECR_REPO)-inference \
		--image-uri $(ECR_URI)/$(ECR_REPO)-inference:latest
	@echo "✅ Deploy completado."

# ============================================================
#  Limpieza
# ============================================================
clean: ## Limpia artefactos generados
	rm -rf $(VENV) __pycache__ .pytest_cache .mypy_cache htmlcov .coverage
	rm -rf data/processed/*.csv models/*.pkl models/*.json
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "🧹 Limpio."
