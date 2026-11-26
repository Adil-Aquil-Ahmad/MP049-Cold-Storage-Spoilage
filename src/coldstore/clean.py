"""Clean a spoilage table using the problems recorded in reports/findings.yaml.

Bad readings become NaN; the model pipeline fills them with the median.
"""
import numpy as np
import pandas as pd
import yaml

from coldstore.config import load_params, resolve


def load_issues(params: dict) -> dict:
    with open(resolve(params["outputs"]["findings"]), encoding="utf-8") as f:
        return yaml.safe_load(f)["data_issues"]


def clean(df: pd.DataFrame, issues: dict, target: str | None = None) -> pd.DataFrame:
    df = df.copy()

    leak = issues["leaking_column"]
    if leak["present"] and leak["column"] in df.columns:
        df = df.drop(columns=leak["column"])

    sentinel = issues["sentinel_value"]
    if sentinel["present"]:
        col = sentinel["column"]
        df[col] = df[col].mask(df[col] == sentinel["value"], np.nan)

    stuck = issues["stuck_sensor"]
    if stuck["present"]:
        col = stuck["column"]
        in_window = df["date"].between(stuck["start_date"], stuck["end_date"])
        stuck_value = df.loc[in_window, col].mode().iloc[0] if in_window.any() else None
        df[col] = df[col].mask(in_window & (df[col] == stuck_value), np.nan)

    unit = issues["unit_mixup"]
    if unit["present"]:
        wrong = df["chamber"] == unit["group_value"]
        df.loc[wrong, unit["column"]] = df.loc[wrong, unit["column"]] / unit["factor"]

    spelling = issues["spelling_variants"]
    if spelling["present"]:
        col = spelling["column"]
        df[col] = df[col].str.strip().str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True)

    if issues["duplicate_rows"]["present"]:
        df = df.drop_duplicates(subset=[c for c in df.columns if c != "record_id"])

    if target and issues["target_typos"]["present"] and target in df.columns:
        too_big = df[target] > 3 * df[target].median()
        df.loc[too_big, target] = df.loc[too_big, target] / 10

    return df.reset_index(drop=True)


def load_clean_train(params: dict | None = None) -> pd.DataFrame:
    params = params or load_params()
    raw = pd.read_csv(resolve(params["data"]["train"]))
    return clean(raw, load_issues(params), params["target"])
