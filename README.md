# MP049-Cold-Storage-Spoilage

SCSE3040 MLOps mini project.

- **Group:** G236
- **Problem:** MP049 — Cold Storage Spoilage (Indore)
- **Faculty:** Upasna Joshi, Batch 38

## Members

| Enrolment no. | Name |
|---|---|
| S24CSEU1664 | Adil Ahmad Aquil |
| S24CSEU1673 | Nishant Kumar |
| S24CSEU1678 | Anushka Mishra |
| S24CSEU1716 | Kanak Chaudhary |

## The problem

Predict `spoilage_pct` (percent of a stored lot found spoiled) for lots of
potato, onion and apple in the Indore Cold Chain Cooperative's cold store.
Regression, measured with MAE.

## Layout

```
data/raw/            train.csv, later.csv (tracked by DVC), holdout_features.csv
params.yaml          paths and model settings, read by src/coldstore/config.py
src/coldstore/       pipeline code (data checks, cleaning, training, monitoring)
app/main.py          FastAPI service: POST /predict, GET /health
tests/               pytest tests
k8s/                 Kubernetes Deployment and Service
reports/findings.yaml
predictions/holdout_predictions.csv
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
pip install dvc==3.67.1        # data versioning, not needed by the service
dvc pull                       # or copy the CSVs from the problem zip into data/raw/
```
