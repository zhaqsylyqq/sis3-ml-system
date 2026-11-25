# SIS3. 23B031056 Abdimutalipov Zhaxylyk
# load registered model from mlflow model registry and predict

import os
import mlflow
from sklearn.datasets import load_wine

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

# load by alias (can also use "models:/wine-classifier/1")
model = mlflow.sklearn.load_model("models:/wine-classifier@champion")

X = load_wine(as_frame=True).data
print("predictions for first 5 rows:", model.predict(X.head()).tolist())
