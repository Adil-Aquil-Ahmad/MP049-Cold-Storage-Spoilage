"""Two months later: drift (PSI), v1 error per chamber, retrain v2, promotion rule.

Run:  python -m coldstore.monitor
"""
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient

from coldstore.clean import clean, load_clean_train, load_issues
from coldstore.config import load_params, resolve
from coldstore.train import data_sha256, fit, mae, setup_mlflow

GROUP_COL = "chamber"


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index with bin edges from the reference (train) deciles."""
    expected, actual = expected.dropna(), actual.dropna()
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def weighted_fit(train: pd.DataFrame, new: pd.DataFrame, weight: float, cfg: dict, params: dict):
    """Fit on train + new rows, giving the new rows `weight` times the influence."""
    rows = pd.concat([train, new], ignore_index=True)
    sw = np.r_[np.ones(len(train)), np.full(len(new), float(weight))]
    return fit(rows, cfg, params, sample_weight=sw)


def choose_weight(train, first, cfg, params) -> tuple[float, dict]:
    """Pick the recency weight on a validation slice inside the first 70% of later.csv."""
    split = int(len(first) * params["monitoring"]["weight_select_from"]
                / params["monitoring"]["retrain_fraction"])
    fit_part, val = first.iloc[:split], first.iloc[split:]
    scores = {w: round(mae(weighted_fit(train, fit_part, w, cfg, params), val, params), 4)
              for w in params["monitoring"]["recency_weights"]}
    best = min(scores, key=scores.get)
    print("recency weight -> validation MAE:", scores, "-> chosen:", best)
    return best, scores


def v1_config(params: dict, run_id: str) -> dict:
    name = MlflowClient().get_run(run_id).data.params["name"]
    return next(c for c in params["model"]["candidates"] if c["name"] == name)


def main() -> None:
    params = load_params()
    setup_mlflow(params)
    client = MlflowClient()
    findings_path = resolve(params["outputs"]["findings"])
    with open(findings_path, encoding="utf-8") as f:
        findings = yaml.safe_load(f)
    model_name = findings["tracking"]["registered_model_name"]
    numeric = params["features"]["numeric"]

    train = load_clean_train(params)
    later = clean(pd.read_csv(resolve(params["data"]["later"])), load_issues(params),
                  params["target"])
    later = later.sort_values(["date", "record_id"]).reset_index(drop=True)

    # 1. drift
    psis = {c: round(psi(train[c], later[c], params["monitoring"]["psi_bins"]), 4)
            for c in numeric}
    most_drifted = max(psis, key=psis.get)
    print("PSI:", psis, "-> most drifted:", most_drifted)

    # 2. v1 error per chamber on later.csv
    v1 = mlflow.sklearn.load_model(f"models:/{model_name}/1")
    by_group = {g: round(mae(v1, part, params), 4)
                for g, part in later.groupby(GROUP_COL)}
    worst = max(by_group, key=by_group.get)
    print("v1 MAE by chamber:", by_group, "-> worst served:", worst)

    # 3. retrain v2 on train + first 70% of later, compare on last 30%
    cut = int(len(later) * params["monitoring"]["retrain_fraction"])
    first, check = later.iloc[:cut], later.iloc[cut:]
    cfg = v1_config(params, findings["tracking"]["best_run_id"])
    mae_v1 = mae(v1, check, params)
    weight, weight_scores = choose_weight(train, first, cfg, params)

    with mlflow.start_run(run_name=f"v2-{cfg['name']}-w{weight}"):
        v2 = weighted_fit(train, first, weight, cfg, params)
        mae_v2 = mae(v2, check, params)
        mlflow.log_param("data_sha256", data_sha256(params))
        mlflow.log_param("model_version", "v2")
        mlflow.log_param("train_rows", len(train) + len(first))
        mlflow.log_param("recency_weight", weight)
        mlflow.log_params(cfg)
        mlflow.log_metrics({f"mae_val_w{w}": s for w, s in weight_scores.items()})
        mlflow.log_metric("mae_check_v1", mae_v1)
        mlflow.log_metric("mae_check_v2", mae_v2)
        mlflow.log_metrics({f"psi_{c}": v for c, v in psis.items()})
        info = mlflow.sklearn.log_model(v2, name="model", serialization_format="cloudpickle")
    v2_version = mlflow.register_model(info.model_uri, model_name).version

    # 4. promotion rule: v2 must be at least 10% better than v1 on the check rows
    needed = params["monitoring"]["promotion_min_improvement"]
    decision = "promote" if mae_v2 <= (1 - needed) * mae_v1 else "keep"
    print(f"check MAE v1 {mae_v1:.4f}, v2 {mae_v2:.4f} ->"
          f" {(1 - mae_v2 / mae_v1):.1%} better, decision: {decision}")
    if decision == "promote":
        client.set_registered_model_alias(model_name, "champion", v2_version)
        joblib.dump(v2, resolve(params["outputs"]["model"]))
        findings["tracking"]["champion_version"] = int(v2_version)

    findings["monitoring"] = {
        "psi": psis,
        "most_drifted_feature": most_drifted,
        "mae_by_group": by_group,
        "worst_served_group": worst,
        "mae_v1_on_check": round(mae_v1, 4),
        "mae_v2_on_check": round(mae_v2, 4),
        "decision": decision,
    }
    with open(findings_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(findings, f, sort_keys=False)


if __name__ == "__main__":
    main()
