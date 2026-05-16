# MLOps Project

ML pipeline with experiment tracking using MLflow.

## What it does
- Trains RandomForest model to predict customer churn
- Tracks experiments, parameters and metrics with MLflow
- Stores best models in MLflow Model Registry

## Stack
- MLflow 2.11
- scikit-learn
- Docker Compose
- Python 3.11

## How to run
1. `docker compose up -d`
2. `conda activate mlops`
3. `python src/train.py`
4. Open localhost:5001 to see experiments