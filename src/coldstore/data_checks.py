"""Look for the seven kinds of data problem in train.csv.

Run:  python -m coldstore.data_checks        (from src/, or with src on PYTHONPATH)
It prints what it finds and writes the data_issues part of reports/findings.yaml.
"""
import re

import pandas as pd
import yaml

from coldstore.config import load_params, resolve

SENTINELS = [-1, -99, -999, -9999, 999, 9999]
UNIT_FACTORS = [10, 12, 60, 100, 1000]
GROUP_COL = "chamber"


def numeric_columns(params: dict) -> list[str]:
    return params["features"]["numeric"]


def check_leaking_column(train: pd.DataFrame, holdout: pd.DataFrame, target: str) -> dict:
    """A column missing from holdout (other than the target) that tracks the target."""
    for col in train.columns:
        if col == target or col in holdout.columns:
            continue
        if pd.api.types.is_numeric_dtype(train[col]):
            corr = train[col].corr(train[target])
            print(f"  leak candidate {col}: corr with target = {corr:.4f}")
            if abs(corr) > 0.95:
                return {"present": True, "column": col}
    return {"present": False, "column": None}


def check_sentinel_value(train: pd.DataFrame, cols: list[str]) -> dict:
    for col in cols:
        counts = train[col].value_counts()
        for value in SENTINELS:
            n = int(counts.get(value, 0))
            if n >= 5:
                print(f"  {col} has {value} on {n} rows")
                return {"present": True, "column": col, "value": value, "count": n}
    return {"present": False, "column": None, "value": None, "count": None}


def check_unit_mixup(train: pd.DataFrame, cols: list[str], sentinel: dict) -> dict:
    """Compare each group's median with the other groups' median, column by column."""
    for col in cols:
        values = train[[GROUP_COL, col]]
        if sentinel["present"] and sentinel["column"] == col:
            values = values[values[col] != sentinel["value"]]
        for group in sorted(values[GROUP_COL].unique()):
            mine = values.loc[values[GROUP_COL] == group, col].median()
            rest = values.loc[values[GROUP_COL] != group, col].median()
            if mine == 0 or rest == 0:
                continue
            ratio = mine / rest
            for factor in UNIT_FACTORS:
                for f in (factor, 1 / factor):
                    if abs(ratio / f - 1) < 0.15:
                        print(f"  {col} in {group}: median ratio {ratio:.2f} ~ {f}")
                        return {"present": True, "column": col,
                                "group_value": group, "factor": f}
    print("  no group differs from the others by a unit factor")
    return {"present": False, "column": None, "group_value": None, "factor": None}


def check_duplicate_rows(train: pd.DataFrame) -> dict:
    exact = int(train.duplicated().sum())
    without_id = int(train.drop(columns="record_id").duplicated().sum())
    print(f"  exact duplicates: {exact}, duplicates ignoring record_id: {without_id}")
    n = max(exact, without_id)
    return {"present": n > 0, "count": n if n > 0 else None}


def check_stuck_sensor(train: pd.DataFrame, cols: list[str], min_run: int = 20) -> dict:
    """Longest run of one repeated value in date order, for every numeric column."""
    ordered = train.sort_values(["date", "record_id"]).reset_index(drop=True)
    best = None
    for col in cols:
        run_id = (ordered[col] != ordered[col].shift()).cumsum()
        runs = ordered.groupby(run_id).agg(
            value=(col, "first"), length=(col, "size"),
            start=("date", "first"), end=("date", "last"))
        top = runs.sort_values("length").iloc[-1]
        print(f"  {col}: longest run {top.length} rows of {top.value}")
        if top.length >= min_run and (best is None or top.length > best[1].length):
            best = (col, top)
    if best is None:
        return {"present": False, "column": None, "start_date": None, "end_date": None}
    col, top = best
    print(f"  stuck: {col} = {top.value} from {top.start} to {top.end}")
    return {"present": True, "column": col,
            "start_date": str(top.start), "end_date": str(top.end)}


def check_target_typos(train: pd.DataFrame, target: str) -> dict:
    """Look for a small group of targets far above the rest, behind a big gap."""
    values = train[target].sort_values().to_numpy()
    median = float(pd.Series(values).median())
    gaps = values[1:] - values[:-1]
    i = int(gaps.argmax())
    top_count = len(values) - (i + 1)
    print(f"  biggest gap {gaps[i]:.2f} between {values[i]} and {values[i + 1]};"
          f" {top_count} values above it (median {median:.2f})")
    # typed with an extra zero: the values above the gap are several times the bulk
    if values[i + 1] > 3 * values[i] and top_count < 0.05 * len(values):
        return {"present": True, "count": top_count}
    return {"present": False, "count": None}


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def check_spelling_variants(train: pd.DataFrame, later: pd.DataFrame) -> dict:
    for col in train.select_dtypes(exclude="number").columns:
        if col in ("record_id", "date"):
            continue
        raw = train[col].nunique()
        clean = train[col].map(normalise).nunique()
        print(f"  {col}: {raw} spellings, {clean} after normalising,"
              f" later.csv has {later[col].nunique()}")
        if raw > clean:
            return {"present": True, "column": col, "true_levels": int(clean)}
    return {"present": False, "column": None, "true_levels": None}


def run_checks(params: dict) -> dict:
    train = pd.read_csv(resolve(params["data"]["train"]))
    later = pd.read_csv(resolve(params["data"]["later"]))
    holdout = pd.read_csv(resolve(params["data"]["holdout"]))
    target = params["target"]
    cols = numeric_columns(params)

    print("leaking_column");    leak = check_leaking_column(train, holdout, target)
    print("sentinel_value");    sentinel = check_sentinel_value(train, cols)
    print("unit_mixup");        unit = check_unit_mixup(train, cols, sentinel)
    print("duplicate_rows");    dups = check_duplicate_rows(train)
    print("stuck_sensor");      stuck = check_stuck_sensor(train, cols)
    print("target_typos");      typos = check_target_typos(train, target)
    print("spelling_variants"); spelling = check_spelling_variants(train, later)

    return {
        "leaking_column": leak,
        "unit_mixup": unit,
        "sentinel_value": sentinel,
        "duplicate_rows": dups,
        "stuck_sensor": stuck,
        "target_typos": typos,
        "spelling_variants": spelling,
    }


def write_findings(issues: dict, path) -> None:
    with open(path, encoding="utf-8") as f:
        findings = yaml.safe_load(f)
    findings["data_issues"] = issues
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(findings, f, sort_keys=False)


def main() -> None:
    params = load_params()
    issues = run_checks(params)
    write_findings(issues, resolve(params["outputs"]["findings"]))
    print(yaml.safe_dump(issues, sort_keys=False))


if __name__ == "__main__":
    main()
