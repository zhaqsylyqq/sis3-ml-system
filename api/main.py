# SIS3. 23B031056 Abdimutalipov Zhaxylyk
# fastapi app that serves the best model from mlflow

import json
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Wine Classifier API")

# load model and its info (both created by train.py)
model = joblib.load("model.joblib")
with open("model_info.json") as f:
    model_info = json.load(f)

class_names = ["class_0", "class_1", "class_2"]

# column names in the same order as in training
columns = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline",
]


# input features (example values = first row of the dataset)
class WineFeatures(BaseModel):
    alcohol: float = 14.23
    malic_acid: float = 1.71
    ash: float = 2.43
    alcalinity_of_ash: float = 15.6
    magnesium: float = 127.0
    total_phenols: float = 2.8
    flavanoids: float = 3.06
    nonflavanoid_phenols: float = 0.28
    proanthocyanins: float = 2.29
    color_intensity: float = 5.64
    hue: float = 1.04
    od280_od315_of_diluted_wines: float = 3.92
    proline: float = 1065.0


@app.get("/")
def root():
    return {"message": "ML API is running"}


@app.get("/model-info")
def get_model_info():
    return model_info


@app.post("/predict")
def predict(features: WineFeatures):
    values = list(features.model_dump().values())
    X = pd.DataFrame([values], columns=columns)

    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]

    probabilities = {}
    for i in range(len(class_names)):
        probabilities[class_names[i]] = round(float(proba[i]), 4)

    return {
        "prediction": pred,
        "class_name": class_names[pred],
        "probability": round(float(proba[pred]), 4),
        "probabilities": probabilities,
        "model_version": model_info["model_version"],
    }
