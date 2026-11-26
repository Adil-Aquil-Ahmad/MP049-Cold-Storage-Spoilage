"""Predict holdout_features.csv with the @champion model from the MLflow registry.

Run:  python -m coldstore.predict
"""
import mlflow
import mlflow.sklearn
import pandas as pd

from coldstore.config import load_params, resolve
from coldstore.train import features, setup_mlflow


def main() -> None:
    params = load_params()
    setup_mlflow(params)
    name = params["mlflow"]["registered_model_name"]
    model = mlflow.sklearn.load_model(f"models:/{name}@champion")

    holdout = pd.read_csv(resolve(params["data"]["holdout"]))
    out = pd.DataFrame({"record_id": holdout["record_id"],
                        "prediction": model.predict(features(holdout, params))})
    path = resolve(params["outputs"]["predictions"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"wrote {len(out)} predictions to {path}")


if __name__ == "__main__":
    main()
