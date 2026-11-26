"""Train v1 on cleaned train.csv, track every candidate in MLflow, register the best.

Run:  python -m coldstore.train
"""
import hashlib

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from coldstore.clean import load_clean_train
from coldstore.config import ROOT, load_params, resolve


def data_sha256(params: dict) -> str:
    with open(resolve(params["data"]["train"]), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def make_regressor(cfg: dict, random_state: int):
    kind = cfg["type"]
    if kind == "ridge":
        return Ridge(alpha=cfg["alpha"])
    if kind == "random_forest":
        return RandomForestRegressor(n_estimators=cfg["n_estimators"],
                                     max_depth=cfg["max_depth"],
                                     random_state=random_state, n_jobs=-1)
    if kind == "gradient_boosting":
        return GradientBoostingRegressor(n_estimators=cfg["n_estimators"],
                                         learning_rate=cfg["learning_rate"],
                                         max_depth=cfg["max_depth"],
                                         random_state=random_state)
    raise ValueError(f"unknown model type {kind}")


def build_pipeline(cfg: dict, params: dict) -> Pipeline:
    numeric = params["features"]["numeric"]
    categorical = params["features"]["categorical"]
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([("prep", prep),
                     ("model", make_regressor(cfg, params["model"]["random_state"]))])


def features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    return df[params["features"]["numeric"] + params["features"]["categorical"]]


def fit(df: pd.DataFrame, cfg: dict, params: dict) -> Pipeline:
    pipe = build_pipeline(cfg, params)
    pipe.fit(features(df, params), df[params["target"]])
    return pipe


def mae(model: Pipeline, df: pd.DataFrame, params: dict) -> float:
    return float(mean_absolute_error(df[params["target"]], model.predict(features(df, params))))


def setup_mlflow(params: dict) -> None:
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])


def update_tracking(params: dict, values: dict) -> None:
    path = resolve(params["outputs"]["findings"])
    with open(path, encoding="utf-8") as f:
        findings = yaml.safe_load(f)
    findings["tracking"].update(values)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(findings, f, sort_keys=False)


def main() -> None:
    params = load_params()
    setup_mlflow(params)
    df = load_clean_train(params)
    sha = data_sha256(params)
    folds = KFold(params["model"]["cv_folds"], shuffle=True,
                  random_state=params["model"]["random_state"])

    results = []
    for cfg in params["model"]["candidates"]:
        with mlflow.start_run(run_name=f"v1-{cfg['name']}") as run:
            pipe = build_pipeline(cfg, params)
            scores = -cross_val_score(pipe, features(df, params), df[params["target"]],
                                      cv=folds, scoring="neg_mean_absolute_error")
            pipe = fit(df, cfg, params)
            mlflow.log_param("data_sha256", sha)
            mlflow.log_param("model_version", "v1")
            mlflow.log_param("train_rows", len(df))
            mlflow.log_params({k: v for k, v in cfg.items()})
            mlflow.log_metric("mae_cv", float(scores.mean()))
            mlflow.log_metric("mae_train", mae(pipe, df, params))
            info = mlflow.sklearn.log_model(pipe, name="model",
                                            serialization_format="cloudpickle")
            print(f"{cfg['name']:>16}: mae_cv = {scores.mean():.4f}  run {run.info.run_id}")
            results.append((scores.mean(), run.info.run_id, info.model_uri, pipe))

    best_mae, best_run, best_uri, best_pipe = min(results, key=lambda r: r[0])
    name = params["mlflow"]["registered_model_name"]
    version = mlflow.register_model(best_uri, name).version
    MlflowClient().set_registered_model_alias(name, "champion", version)

    out = resolve(params["outputs"]["model"])
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, out)

    update_tracking(params, {
        "experiment_name": params["mlflow"]["experiment_name"],
        "best_run_id": best_run,
        "registered_model_name": name,
        "champion_version": int(version),
    })
    print(f"best run {best_run} (mae_cv {best_mae:.4f}) registered as {name} v{version},"
          f" alias champion; saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
