# GridPulse: Smart Energy Demand Forecasting and Optimization

An end-to-end MLOps system that predicts hourly electricity demand, with automated data validation, training, deployment and monitoring.

**Data:** hourly demand from the [U.S. EIA Open Data API](https://www.eia.gov/opendata/) (California ISO) plus hourly weather from [Open-Meteo](https://open-meteo.com/).
**Models:** XGBoost, LightGBM, Random Forest, with a seasonal baseline for comparison.
**Why it matters:** time-based validation, changing seasonal patterns, and measurable prediction error make it a realistic forecasting problem.

## Team

| Student | Name | Primary role |
|---|---|---|
| 1 | Sami Shah | Data and ML engineering |
| 2 | _TBD_ | MLOps and automation |
| 3 | _TBD_ | Deployment and reliability |

## Setup

Requires **Python 3.12**. Some pinned packages have no Python 3.13 builds.

```bash
git clone https://github.com/sami-shah-210504/GridPulse.git
cd GridPulse

# Create the environment (uv downloads Python 3.12 if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 --seed .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env               # then add your EIA API key
```

If you already have Python 3.11 or 3.12, `python3 -m venv .venv` works instead of `uv`.

### Environment variables

| Name | Purpose |
|---|---|
| `EIA_API_KEY` | Free key from https://www.eia.gov/opendata/ (required) |
| `MLFLOW_TRACKING_URI` | MLflow server address (used from the experiment-tracking step) |
| `AWS_REGION` | AWS region (used from the deployment step) |

Never commit `.env`. It is in `.gitignore`.

## Usage

Run all commands from the repo root with the virtual environment active.

```bash
python -m src.data.ingestion     # download raw demand + weather -> data/raw/energy_raw.csv
python -m src.data.validation    # validate the raw data, writes reports/validation_report.json
```

Both commands read their settings from `configs/pipeline_config.yaml` (region, date range, weather location, file paths). Validation exits with an error if the data fails a mandatory check, so a bad dataset never reaches training.

### Data validation checks

- Schema: expected columns, types, unique timestamps, no unexpected columns
- Plausible ranges (for example humidity 0 to 100, demand above 0)
- Missing values: at most 5% null per column
- Time continuity: no gap longer than 6 hours in the hourly series

## Project structure

```
configs/             Pipeline and model settings (YAML)
data/raw/            Raw datasets (tracked by DVC, not Git)
data/processed/      Feature-engineered data (tracked by DVC, not Git)
src/data/            Ingestion and validation
src/features/        Preprocessing and feature engineering
src/models/          Training, tuning, evaluation
src/serving/         API code
src/monitoring/      Drift and performance monitoring
src/utils/           Config loading and logging
tests/               Automated tests
pipelines/           Airflow DAGs and Kubeflow pipelines
feature_store/       Feast feature repository
docker/, k8s/        Container and Kubernetes definitions
.github/workflows/   CI/CD
```

Some folders are still empty. They get filled as each part of the project is built.

## Team workflow

- Nobody commits directly to `main`. It is protected and requires a pull request.
- Create a branch for each piece of work: `git checkout -b feature/<name>`.
- Use clear commit prefixes: `feat:`, `fix:`, `chore:`, `docs:`, `test:`.
- Every PR is reviewed by another team member before merging.

## Project status

- [x] Project setup and repository structure
- [x] Data ingestion and validation
- [ ] DVC data versioning and DagsHub
- [ ] Preprocessing and feature engineering
- [ ] Feast feature store
- [ ] Training, tuning and MLflow tracking
- [ ] Automated tests
- [ ] Flask API and BentoML serving
- [ ] Docker and Docker Compose
- [ ] GitHub Actions CI/CD
- [ ] Airflow and Kubeflow pipelines
- [ ] Kubernetes deployment
- [ ] AWS deployment
- [ ] Monitoring (Prometheus, Grafana, Evidently)
- [ ] Automated retraining, promotion and rollback