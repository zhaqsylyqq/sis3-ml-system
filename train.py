# SIS3. 23B031056 Abdimutalipov Zhaxylyk
# train several models, track them in mlflow, register the best one

import os
import json
import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

SEED = 42
EXPERIMENT_NAME = "wine-classification"
MODEL_NAME = "wine-classifier"

# mlflow storage (local sqlite db by default)
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(EXPERIMENT_NAME)
print("tracking uri:", tracking_uri)

# load data
data = load_wine(as_frame=True)
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# models we want to compare
models = [
    ("logreg", Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=1.0, max_iter=1000))]),
     {"model_type": "LogisticRegression", "C": 1.0, "max_iter": 1000, "scaling": "standard"}),
    ("knn", Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]),
     {"model_type": "KNeighborsClassifier", "n_neighbors": 5, "scaling": "standard"}),
    ("decision_tree", DecisionTreeClassifier(max_depth=4, random_state=SEED),
     {"model_type": "DecisionTreeClassifier", "max_depth": 4}),
    ("random_forest_100", RandomForestClassifier(n_estimators=100, random_state=SEED),
     {"model_type": "RandomForestClassifier", "n_estimators": 100, "max_depth": "None"}),
    ("random_forest_200_d5", RandomForestClassifier(n_estimators=200, max_depth=5, random_state=SEED),
     {"model_type": "RandomForestClassifier", "n_estimators": 200, "max_depth": 5}),
]

# knn and tree models use these inner types, mlflow (skops format) needs to trust them
trusted_types = [
    "sklearn.metrics._dist_metrics.EuclideanDistance64",
    "sklearn.neighbors._kd_tree.KDTree",
    "sklearn.tree._tree.Tree",
]

results = []

for run_name, model, params in models:
    with mlflow.start_run(run_name=run_name) as run:
        # 5-fold cv on train part
        cv_f1 = float(cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro").mean())

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "cv_f1_macro": cv_f1,
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_f1_macro": f1_score(y_test, y_pred, average="macro"),
            "test_precision_macro": precision_score(y_test, y_pred, average="macro"),
            "test_recall_macro": recall_score(y_test, y_pred, average="macro"),
        }

        # log params and metrics
        mlflow.log_params(params)
        mlflow.log_param("seed", SEED)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_metrics(metrics)

        # log model artifact
        signature = infer_signature(X_train, model.predict(X_train))
        logged = mlflow.sklearn.log_model(
            model, name="model", signature=signature, input_example=X_train.head(3),
            skops_trusted_types=trusted_types,
        )

        # also log joblib file
        joblib.dump(model, "model_tmp.joblib")
        mlflow.log_artifact("model_tmp.joblib", artifact_path="joblib")
        os.remove("model_tmp.joblib")

        print(run_name, {k: round(v, 4) for k, v in metrics.items()})
        results.append((run_name, model, metrics, run.info.run_id, logged.model_uri))

# choose best model: first by cv f1, then by test f1
best = results[0]
for r in results:
    if (r[2]["cv_f1_macro"], r[2]["test_f1_macro"]) > (best[2]["cv_f1_macro"], best[2]["test_f1_macro"]):
        best = r
best_name, best_model, best_metrics, best_run_id, best_uri = best
print("\nbest model:", best_name, "| cv f1:", round(best_metrics["cv_f1_macro"], 4))

# register best model in model registry
mv = mlflow.register_model(best_uri, MODEL_NAME)
client = MlflowClient()
client.update_model_version(
    name=MODEL_NAME,
    version=mv.version,
    description=f"{best_name} trained on wine dataset, cv f1 = {best_metrics['cv_f1_macro']:.4f}",
)
client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
client.set_model_version_tag(MODEL_NAME, mv.version, "algorithm", best_name)
print(f"registered: {MODEL_NAME} version {mv.version} (alias: champion)")

# save model for the api
joblib.dump(best_model, "api/model.joblib")

info = {
    "model_name": MODEL_NAME,
    "model_version": int(mv.version),
    "algorithm": best_name,
    "run_id": best_run_id,
    "metrics": {k: round(v, 4) for k, v in best_metrics.items()},
}
with open("api/model_info.json", "w") as f:
    json.dump(info, f, indent=2)
print("saved api/model.joblib and api/model_info.json")
