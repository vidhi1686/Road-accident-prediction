# Accident Severity Classification with MLOps

**Course:** CSL236 – Introduction to AI and Machine Learning, B.Tech CSE, Semester 5, The NorthCap University
**Project type:** End-term MLOps system (reproducible preprocessing → tracked experiments → validated API → tests → Docker → CI → deployment configuration)

| Team member | Roll number | Contributions |
| --- | --- | --- |
| Vidhi Jindal | 24csu324 | _to be confirmed by the team_ |
| Palak Khanna | 24csu325 | _to be confirmed by the team_ |

---

## 1. Project objective

This project classifies the **severity of a reported road accident** from
accident-report information. It is a **binary classification** system:

- `Non-severe` (0): the report records a slight injury.
- `Severe` (1): the report records a serious or fatal injury.

The service returns a calibrated-style probability, a human-readable label and
2–3 rule-based safety tips. It is exposed as a FastAPI service with a browser
UI, tracked with MLflow, tested with pytest, containerised with Docker and
prepared for Render deployment.

> **Important scope statement:** this is accident-severity classification using
> accident-report information. It is **not** a pre-accident crash predictor.
> Some inputs, such as the number of casualties and the collision type, are
> recorded only after the accident has occurred.

## 2. Dataset

| Item | Value |
| --- | --- |
| Raw file | `data/raw/Road.csv` |
| Cleaned file | `data/accidents.csv` |
| Original shape | 12,316 rows × 32 columns |
| Cleaned shape | 12,316 rows × 16 columns (15 features + target) |
| Source | Kaggle — [Road Accident Severity in India](https://www.kaggle.com/datasets/s3programmer/road-accident-severity-in-india) |
| License | Not included in this repository. Refer to the Kaggle dataset page for the dataset's terms. |

The raw file is the original Kaggle CSV supplied for the project. The
repository does **not** redistribute a synthetic replacement; `data/raw/Road.csv`
is the source of truth and `data/accidents.csv` is generated deterministically
from it.

### Original target distribution

| `Accident_severity` | Rows | Share |
| --- | ---: | ---: |
| Slight Injury | 10,415 | 84.56% |
| Serious Injury | 1,743 | 14.15% |
| Fatal injury | 158 | 1.28% |
| **Total** | **12,316** | **100%** |

### Binary target mapping

| Original value | `severe` |
| --- | ---: |
| Slight Injury | 0 (Non-severe) |
| Serious Injury | 1 (Severe) |
| Fatal injury | 1 (Severe) |

| `severe` | Rows | Share |
| --- | ---: | ---: |
| 0 — Non-severe | 10,415 | 84.5648% |
| 1 — Severe | 1,901 | 15.4352% |

The dataset is imbalanced. Accuracy alone is therefore not used to judge or
select a model; macro F1, severe-class recall, severe-class precision and
ROC-AUC are reported alongside it.

## 3. Curated feature contract

The model uses 15 features that are available in an accident report. The
contract is defined once in `schema.py`.

### Numeric (4)

| Feature | Source | Description |
| --- | --- | --- |
| `hour` | derived from `Time` | Hour of day, 0–23 |
| `is_weekend` | derived from `Day_of_week` | 1 for Saturday/Sunday, else 0 |
| `num_vehicles` | `Number_of_vehicles_involved` | Vehicles involved |
| `num_casualties` | `Number_of_casualties` | Reported casualties |

### Ordinal (2)

| Feature | Source | Verified order |
| --- | --- | --- |
| `age_band_of_driver` | `Age_band_of_driver` | Under 18 → 18-30 → 31-50 → Over 51 → Unknown |
| `driving_experience` | `Driving_experience` | No Licence → Below 1yr → 1-2yr → 2-5yr → 5-10yr → Above 10yr → Unknown |

### Nominal (9)

`sex_of_driver`, `type_of_vehicle`, `area_accident_occurred`,
`types_of_junction`, `road_surface_type`, `road_surface_conditions`,
`light_conditions`, `weather_conditions`, `type_of_collision`.

The exact category values are derived from and verified against the cleaned data,
stored in `schema.CATEGORY_VALUES`, and served to the frontend by `/categories`
so the UI and backend cannot drift.

### Leakage prevention

The following casualty/outcome columns are **excluded** because they are
recorded after the accident and would leak the target:

`Casualty_severity`, `Casualty_class`, `Sex_of_casualty`,
`Age_band_of_casualty`, `Work_of_casuality`, `Fitness_of_casuality`,
`Pedestrian_movement`.

Other columns outside the curated contract are also excluded, including
`Educational_level`, `Vehicle_driver_relation`, `Owner_of_vehicle`,
`Service_year_of_vehicle`, `Defect_of_vehicle`, `Lanes_or_Medians`,
`Road_allignment`, `Vehicle_movement` and `Cause_of_accident`.

`Casualty_severity` is a particularly direct leak: it is a casualty-level
severity assessment, not a pre-report feature.

### Missing values

The cleaned dataset preserves missing values as real `NaN`s; no rows are
dropped. The observed missingness is:

| Feature | Missing | Share |
| --- | ---: | ---: |
| `type_of_vehicle` | 950 | 7.71% |
| `driving_experience` | 829 | 6.73% |
| `types_of_junction` | 887 | 7.20% |
| `area_accident_occurred` | 239 | 1.94% |
| `road_surface_type` | 172 | 1.40% |
| `type_of_collision` | 155 | 1.26% |

Imputation is **not** performed in the preprocessing script. Median and
most-frequent imputers live inside the scikit-learn pipeline and are fitted on
the training split only.

## 4. Reproducible preprocessing

`data/preprocess.py` transforms the raw CSV into the modelling dataset:

1. Validates the required source columns.
2. Derives `hour` from `Time` and `is_weekend` from `Day_of_week`.
3. Maps the three-class target to binary `severe`.
4. Selects and renames the 15 approved features.
5. Trims whitespace, merges whitespace variants, unifies `unknown`/`Unknown`,
   and converts empty/`na`/`nan` tokens to real missing values.
6. Prints shape, target distribution and missing-value percentages.
7. Writes `data/accidents.csv`.

```bash
python data/preprocess.py
python data/preprocess.py --input data/raw/Road.csv --output data/accidents.csv
```

Output: **12,316 rows × 16 columns**, with 1,901 severe cases (15.4352%). The
script asserts that the row count is preserved and that no leakage column is
present.

The original raw CSV is never modified.

## 5. Training and model selection

Run:

```bash
python train.py --all --promote always
```

`train.py` creates one stratified **train / validation / test** split and
reuses it for every candidate:

| Split | Rows | Share |
| --- | ---: | ---: |
| Train | 7,881 | 64% |
| Validation | 1,971 | 16% |
| Test | 2,464 | 20% |

`random_state=42`, `test_size=0.20`, `validation_size=0.20` (the validation
share is taken from the 80% train+validation pool).

### Pipeline

- Numeric: median imputation → `RobustScaler`.
- Ordinal: most-frequent imputation → `OrdinalEncoder` with explicit category
  ordering, `handle_unknown="use_encoded_value"`, `unknown_value=-1`.
- Nominal: most-frequent imputation → `OneHotEncoder(handle_unknown="ignore")`.
- The complete preprocessing-and-estimator pipeline is fitted on training data
  only, preventing leakage.

Class imbalance handling:

- Logistic Regression: `class_weight="balanced"`.
- Random Forest: `class_weight="balanced_subsample"`.
- Gradient Boosting: balanced sample weights passed through the pipeline.

The notebook's `VarianceThreshold` + `SelectKBest(f_classif, k=50)` is **not
retained**. The curated contract produces about 85 one-hot columns rather than
the notebook's 183, so the dimensionality is already manageable; `SelectKBest`
can discard rare but valid categories and complicates feature-importance
reporting. Numeric outliers are retained because extreme vehicle/casualty
counts are plausible accident reports, and `RobustScaler` limits their
influence.

### Six experiments — validation results

All candidates use the same split. Selection uses validation data only; the
test set is not touched.

| Run | Algorithm | Hyperparameters | Val macro F1 | Val severe recall | Val precision | Val ROC-AUC | Selected |
| --- | --- | --- | ---: | ---: | ---: | ---: | :---: |
| `rf-n100-d6` | Random Forest | `n_estimators=100, max_depth=6` | 0.5996 | **0.3914** | 0.3020 | 0.6596 | **Yes** |
| `rf-n200-d10` | Random Forest | `n_estimators=200, max_depth=10` | **0.6076** | 0.3125 | 0.3493 | 0.6702 | No |
| `gb-n100-d3-lr0.1` | Gradient Boosting | `n_estimators=100, max_depth=3, lr=0.1` | 0.5769 | 0.4770 | 0.2617 | **0.6764** | No |
| `gb-n200-d2-lr0.05` | Gradient Boosting | `n_estimators=200, max_depth=2, lr=0.05` | 0.5703 | **0.4803** | 0.2539 | 0.6613 | No |
| `logreg-C1.0` | Logistic Regression | `C=1.0` | 0.4947 | 0.5724 | 0.1973 | 0.6037 | No |
| `logreg-C0.1` | Logistic Regression | `C=0.1` | 0.4910 | 0.5526 | 0.1927 | 0.6014 | No |

**Selection rule:** maximise validation macro F1; among candidates within 0.01
of the best, prefer higher severe-class recall, then precision. `rf-n200-d10`
had the highest macro F1 (0.6076), but `rf-n100-d6` (0.5996) was within the
0.01 tolerance and had materially higher severe recall (0.3914 vs 0.3125), so
it was selected. This is a deliberate recall-oriented trade-off for a
severity-screening tool.

**Decision threshold:** tuned on validation predictions only (severe-class F1,
tie-broken by recall). The selected threshold is **0.50**.

The selected pipeline was then refitted on train + validation and evaluated
**once** on the held-out test set.

### Final held-out test results

| Metric | Value |
| --- | ---: |
| Accuracy | 0.7593 |
| Severe precision | 0.2940 |
| Severe recall | 0.4000 |
| Severe F1 | 0.3389 |
| Macro F1 | 0.5959 |
| ROC-AUC | 0.6443 |

Confusion matrix (rows = actual, columns = predicted):

|  | Predicted Non-severe | Predicted Severe |
| --- | ---: | ---: |
| Actual Non-severe | 1,719 | 365 |
| Actual Severe | 228 | 152 |

Severe-class report: precision 0.2940, recall 0.4000, F1 0.3389, support 380.
Macro-average report: precision 0.5885, recall 0.6124, F1 0.5959, support 2,464.

**Interpretation:** accuracy is 75.9%, but that is dominated by the 84.6%
non-severe majority. The severe class remains difficult: the model detects 40%
of severe cases with 29.4% precision. These are honest out-of-sample results;
no target metric is guaranteed. The model should be treated as a screening aid,
not an automated triage decision.

## 6. MLflow experiment tracking

All six candidates are logged to the new experiment
**`road-accident-severity-real`** in the SQLite store `mlflow.db`. The old
synthetic-data experiment is not reused, so real and synthetic runs are never
mixed.

Each candidate run logs:

- Algorithm and hyperparameters.
- Dataset name, row count, SHA-256 content hash and severe prevalence.
- Train/validation/test sizes, random state and split fractions.
- Validation accuracy, precision, recall, F1, macro F1 and ROC-AUC.
- The candidate pipeline artifact under `model/`.

The selected run additionally logs the decision threshold, tuned validation
metrics, final test metrics, the promoted model artifact and `metrics.json`.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
# open http://127.0.0.1:5001
```

`mlruns/` and `mlartifacts/` are gitignored.

Fresh evidence: [MLflow runs](docs/screenshots/mlflow_runs.png) and the
[selected run detail](docs/screenshots/mlflow_run_detail.png).

## 7. API

The FastAPI service loads the complete pipeline once at process start.

| Endpoint | Method | Description |
| --- | --- | --- |
| `/` | GET | Prediction UI (`static/index.html`) |
| `/health` | GET | Liveness probe → `{"status":"ok","model_loaded":true}` |
| `/categories` | GET | Allowed category values from `schema.CATEGORY_VALUES` |
| `/model-info` | GET | Algorithm, test metrics, threshold and top feature importances |
| `/predict` | POST | Validated severity prediction + probability + tips |
| `/docs` | GET | Interactive OpenAPI documentation |

Request validation uses Pydantic:

- Missing fields → HTTP 422.
- Extra fields (`extra="forbid"`) → HTTP 422.
- Unknown categories → HTTP 422.
- Invalid numeric values (e.g. hour 24, weekend 2, negative casualties) → HTTP 422.
- Numeric bounds are application-level sanity bounds, not dataset-observation
  limits.

Example request (a real high-confidence row from the cleaned data):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "hour": 23,
    "is_weekend": 0,
    "num_vehicles": 1,
    "num_casualties": 4,
    "age_band_of_driver": "Under 18",
    "driving_experience": "5-10yr",
    "sex_of_driver": "Male",
    "type_of_vehicle": "Other",
    "area_accident_occurred": "Other",
    "types_of_junction": "No junction",
    "road_surface_type": "Asphalt roads",
    "road_surface_conditions": "Dry",
    "light_conditions": "Darkness - no lighting",
    "weather_conditions": "Normal",
    "type_of_collision": "Vehicle with vehicle collision"
  }'
```

```json
{
  "prediction": 1,
  "label": "Severe",
  "severity_probability": 0.7982,
  "tips": [
    "Low visibility: use headlights and watch for pedestrians and cyclists.",
    "Multiple casualties reported: prioritise emergency response and secure the scene.",
    "Model flags this report as high severity: prioritise emergency response."
  ]
}
```

Every prediction is appended to `logs/predictions.jsonl` with the timestamp,
all 15 inputs, prediction, label, probability and tips. A 20-record sample of
real API predictions is committed at `docs/sample_predictions.jsonl`.

## 8. Frontend

`static/index.html` provides the browser UI:

- 15 inputs matching the contract exactly.
- Categorical dropdowns populated from `/categories`, so the frontend cannot
  diverge from the backend schema.
- Numeric inputs for hour, vehicles and casualties; a weekend selector.
- Lower-risk and higher-risk example buttons using valid categories.
- Binary verdict, severe-probability meter, safety tips and top feature
  importances.
- Clear display of FastAPI validation errors.

Fresh evidence: [prediction UI](docs/screenshots/ui_prediction.png) and
[API docs](docs/screenshots/api_docs.png).

## 9. Automated tests

```bash
pytest -v --tb=short
```

**27 tests pass.**

`tests/test_app.py` covers:

- Health endpoint.
- Frontend loading and all 15 named feature fields.
- `/categories` contract.
- Valid prediction response and probability range.
- Binary response structure and 2–3 tips.
- Missing fields and extra fields → 422.
- Invalid categories and invalid numeric values → 422.
- Boundary values accepted.
- `/model-info`.
- Prediction logging.

`tests/test_preprocess.py` covers:

- Target mapping and counts (10,415 non-severe / 1,901 severe).
- Expected 12,316 × 16 output shape and row preservation.
- Output columns and binary target.
- Severe prevalence.
- Absence of leakage columns.
- Schema compatibility with `schema.py`.
- Missing-value handling and unknown/whitespace normalisation.
- Numeric domains.

Captured output: `docs/pytest_output.txt`.

## 10. GitHub Actions CI

`.github/workflows/ci.yml` runs one job so the Docker build receives the newly
trained model from the same workspace:

1. Check out the repository.
2. Install `requirements.txt` (Python 3.11).
3. Run `python data/preprocess.py`.
4. Validate the processed dataset shape, columns and binary target.
5. Reset the CI MLflow store.
6. Train all six experiments and promote the selected model
   (`python train.py --all --promote always`).
7. Run `pytest -v --tb=short`.
8. Build the Docker image.
9. Start the container, poll `/health`, then stop and remove it cleanly.

The Docker step does not consume a stale committed model: the training step
runs in the same job and writes the schema-compatible `model/model.joblib`
before the image is built.

## 11. Docker

The serving image contains only `app.py`, `schema.py`, `model/`, `static/` and
`requirements-serve.txt`. It does **not** include the raw or processed dataset.

```bash
docker build -t road-accident-prediction .
docker run -d -p 8000:8000 --restart unless-stopped road-accident-prediction
```

Verified locally:

- Image size: **152.7 MB**.
- `/health` returned `{"status":"ok","model_loaded":true}`.
- `/predict` returned a valid binary response.
- The container was stopped and removed after verification.

The server honours Render's `PORT` environment variable with `8000` as the
local fallback:

```dockerfile
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
```

## 12. Render deployment configuration

`render.yaml` remains a one-click Docker blueprint:

```yaml
services:
  - type: web
    name: road-accident-prediction
    runtime: docker
    plan: free
    healthCheckPath: /health
    autoDeploy: true
```

Render sets `PORT`; the container binds to it. The health-check path remains
`/health`.

**No deployment was performed as part of this rebuild.** Commit and push the
new model artifact (`model/model.joblib`) and `model/metrics.json` before
creating or redeploying the Render service, because the image serves the
committed model rather than training at startup.

## 13. Drift monitoring

```bash
python drift_check.py
```

`drift_check.py` uses `schema.NUMERIC_FEATURES` and
`schema.CATEGORICAL_FEATURES`, so it follows the new contract automatically. It
compares prediction-log inputs against `data/accidents.csv`:

- Numeric features: shift in training standard deviations (default flag 0.5σ).
- Categorical features: total-variation distance (default flag 0.2 TVD).
- Exits with status 1 if any feature crosses its threshold.

A verification run with 300+ representative API predictions reported no
meaningful drift. With fewer than 30 logged predictions the tool warns that
estimates are noisy.

**Render limitation:** Render's local filesystem is ephemeral. The
`logs/predictions.jsonl` file can be lost on restart or redeploy, so production
drift monitoring needs persistent storage or external log shipping.

## 14. Repository layout

```
Road-accident-prediction/
├── data/
│   ├── raw/Road.csv             # original Kaggle CSV (committed)
│   ├── accidents.csv            # generated 12,316 × 16 cleaned dataset
│   └── preprocess.py            # deterministic preprocessing
├── model/
│   ├── model.joblib             # selected fitted pipeline
│   └── metrics.json             # selected model + all candidate validation results
├── notebooks/
│   └── data_preprocessing_updated(1) (1).ipynb  # original academic notebook
├── static/
│   └── index.html               # browser UI
├── tests/
│   ├── test_app.py              # API contract, validation, logging
│   └── test_preprocess.py       # preprocessing and schema compatibility
├── docs/
│   ├── pytest_output.txt
│   ├── sample_predictions.jsonl
│   └── screenshots/              # UI, API docs, MLflow runs
├── .github/workflows/ci.yml
├── app.py                       # FastAPI service
├── schema.py                    # single source of truth for features/categories
├── train.py                     # six experiments + validation-based selection
├── drift_check.py               # input-drift monitor
├── Dockerfile
├── requirements.txt
├── requirements-serve.txt
├── pytest.ini
├── render.yaml
├── mlflow.db                    # six real-data MLflow runs
└── README.md
```

The obsolete synthetic generator `data/generate_data.py` and its old artifacts
have been removed.

## 15. Local setup and commands

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 1. Build the cleaned dataset
python data/preprocess.py

# 2. Train all six candidates, select on validation, evaluate test once
python train.py --all --promote always

# 3. Explore experiments
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
# open http://127.0.0.1:5001

# 4. Run tests
pytest -v --tb=short

# 5. Serve
uvicorn app:app --reload --port 8000
# UI:      http://127.0.0.1:8000/
# API docs: http://127.0.0.1:8000/docs
```

Run an individual experiment if needed:

```bash
python train.py --algorithm rf --n_estimators 100 --max_depth 6
python train.py --algorithm rf --n_estimators 200 --max_depth 10
python train.py --algorithm gb --n_estimators 100 --max_depth 3 --learning_rate 0.1
python train.py --algorithm gb --n_estimators 200 --max_depth 2 --learning_rate 0.05
python train.py --algorithm logreg --C 1.0
python train.py --algorithm logreg --C 0.1
```

Promotion controls:

- `--promote auto` (default): promote only if validation macro F1 beats the
  existing real-data metrics. Old synthetic metrics are ignored.
- `--promote always`: overwrite `model/model.joblib` and `model/metrics.json`.
- `--promote never`: generate and log a candidate without promoting it.

## 16. Original notebook

The original academic notebook is preserved at
`notebooks/data_preprocessing_updated(1) (1).ipynb`. It documents the
three-class exploratory workflow: missing-value analysis, time binning,
ordinal/nominal encoding, `VarianceThreshold` + `SelectKBest(k=50)`, and a
three-class Logistic Regression baseline.

Deliberate differences in the production rebuild:

| Notebook | Production rebuild |
| --- | --- |
| Three classes (Slight / Serious / Fatal) | Binary `severe` (Serious + Fatal = 1) |
| 31 raw features, including casualty-outcome columns | 15 curated report-time features, leakage excluded |
| 183 encoded features, `SelectKBest(k=50)` | ~85 encoded features, no selector |
| 80/20 train/test | 64/16/20 train/validation/test |
| Test-set model inspection | Validation-only selection; one final test evaluation |
| Logistic Regression only | RF / GB / LogReg candidates |
| No tracking, persistence or serving | MLflow, Joblib, FastAPI, tests, Docker, CI |

## 17. Limitations

- The model is trained on Indian city-police accident records and may not
  generalise to other countries, reporting systems or time periods.
- The severe class is imbalanced (15.4%); severe recall is 40% and severe
  precision is 29.4% on the held-out test set. Accuracy alone is misleading.
- Some selected inputs are report-time or post-collision information
  (`num_casualties`, `type_of_collision`). This is severity classification, not
  pre-accident prediction.
- No temporal validation, probability calibration or cost-sensitive threshold
  analysis has been performed beyond the validation-tuned threshold.
- Render's filesystem is ephemeral, so prediction logs may not survive restarts.
- The Kaggle dataset page should be consulted for licensing terms; this
  repository does not grant a separate dataset license.
- The model is a screening aid, not a substitute for emergency services or
  human judgement.
