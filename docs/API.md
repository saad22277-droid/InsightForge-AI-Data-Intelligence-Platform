# API reference

Generated from the running application's OpenAPI schema (`python scripts/generate_api_docs.py`).
Interactive documentation is always available at **http://127.0.0.1:8000/docs** (Swagger UI) and `/redoc`.

Conventions: JSON in/out; errors use `{"detail": ...}`. AI errors use `{"detail": {"code": "quota|rate_limited|timeout|auth|not_configured|...", "message": "..."}}` with HTTP 429/502/503/504 and a `Retry-After` header where relevant. Every response carries `X-Request-ID`.

## Datasets (Stages 1-4, unchanged contracts)

| Method | Path | Request body | Query parameters | Description |
|---|---|---|---|---|
| GET | `/datasets/` |  |  | List Datasets |
| DELETE | `/datasets/clear` |  |  | Clear Workspace |
| POST | `/datasets/upload` | `multipart/form-data` |  | Upload Dataset |
| DELETE | `/datasets/{dataset_id}` |  |  | Delete Dataset |
| GET | `/datasets/{dataset_id}` |  |  | Get Dataset |
| GET | `/datasets/{dataset_id}/export` |  |  | Export Dataset |
| GET | `/datasets/{dataset_id}/profile` |  |  | Get Dataset Profile |
| GET | `/datasets/{dataset_id}/quality` |  |  | Get Dataset Quality |
| POST | `/datasets/{dataset_id}/query` | `DatasetQueryRequest` |  | Query Dataset Endpoint |
| GET | `/datasets/{dataset_id}/summary` |  |  | Get Dataset Summary |
| POST | `/datasets/{dataset_id}/validate` | `DatasetValidationRequest` |  | Validate Dataset Endpoint |
| GET | `/datasets/{dataset_id}/versions` |  |  | List Dataset Versions |
| GET | `/datasets/{dataset_id}/versions/compare` |  | from, to | Compare Dataset Versions |
| POST | `/datasets/{dataset_id}/versions/upload` | `multipart/form-data` |  | Upload Dataset Version |
| GET | `/datasets/{dataset_id}/visualization` |  | chart_type, x_column, y_column, bins | Get Dataset Visualization |

## Analysis, charts and exports

| Method | Path | Request body | Query parameters | Description |
|---|---|---|---|---|
| GET | `/datasets/{dataset_id}/charts` |  | chart_type, x_column, y_column, group_by, agg, bins, columns | Get Chart |
| GET | `/datasets/{dataset_id}/eda` |  |  | Complete automated exploratory analysis (computed from the data). |
| GET | `/datasets/{dataset_id}/exports/columns.csv` |  |  | Per-column profile as CSV (counts, missing, statistics, outliers). |
| GET | `/datasets/{dataset_id}/metadata` |  |  | Get Metadata |
| GET | `/datasets/{dataset_id}/versions/{version_number}/download` |  |  | Download Version |

## Data cleaning and dataset versions

| Method | Path | Request body | Query parameters | Description |
|---|---|---|---|---|
| POST | `/datasets/{dataset_id}/cleaning/apply` | `CleaningApplyRequest` |  | Apply operations and store the result as a NEW dataset version. |
| GET | `/datasets/{dataset_id}/cleaning/history` |  |  | History |
| POST | `/datasets/{dataset_id}/cleaning/preview` | `CleaningRequest` |  | Dry run: shows before/after comparison. Nothing is saved. |
| GET | `/datasets/{dataset_id}/cleaning/recommendations` |  |  | Recommendations |
| POST | `/datasets/{dataset_id}/versions/{version_number}/activate` |  |  | Make an earlier version the active one (rollback). No data is deleted. |

## Machine learning studio

| Method | Path | Request body | Query parameters | Description |
|---|---|---|---|---|
| POST | `/datasets/{dataset_id}/ml/compare` | `CompareRequest` |  | Train several algorithms with identical settings; ranked by primary metric. |
| GET | `/datasets/{dataset_id}/ml/runs` |  | task | List Runs |
| POST | `/datasets/{dataset_id}/ml/train` | `TrainRequest` |  | Train |
| GET | `/ml/catalog` |  |  | Catalog |
| DELETE | `/ml/runs/{run_id}` |  |  | Delete Run |
| GET | `/ml/runs/{run_id}` |  |  | Get Run |
| POST | `/ml/runs/{run_id}/predict` | `PredictRequest` |  | Predict |
| POST | `/ml/runs/{run_id}/reproduce` |  |  | Reproduce |
| GET | `/ml/runs/{run_id}/schema` |  |  | Run Schema |

## Experiment tracking

| Method | Path | Request body | Query parameters | Description |
|---|---|---|---|---|
| GET | `/experiments` |  | task, dataset_id, limit | Experiment tracker: every training run across datasets (newest first). |
| GET | `/experiments/compare` |  | ids | Compare |
| GET | `/experiments/summary` |  |  | Summary |

## Gemini AI: chat, explanations, status

| Method | Path | Request body | Query parameters | Description |
|---|---|---|---|---|
| GET | `/ai/status` |  | check | AI service status. ``check=true`` makes one tiny live Gemini call. |
| POST | `/datasets/{dataset_id}/ai/explain` | `ExplainRequest` |  | Explain Endpoint |
| POST | `/datasets/{dataset_id}/chat` | `ChatRequest` |  | Chat |
| GET | `/datasets/{dataset_id}/conversations` |  |  | List Conversations |
| DELETE | `/datasets/{dataset_id}/conversations/{conversation_id}` |  |  | Delete Conversation |
| GET | `/datasets/{dataset_id}/conversations/{conversation_id}` |  |  | Get Conversation |

## Reports

| Method | Path | Request body | Query parameters | Description |
|---|---|---|---|---|
| GET | `/datasets/{dataset_id}/reports` |  |  | List Reports |
| POST | `/datasets/{dataset_id}/reports` | `ReportRequest` |  | Create Report |
| DELETE | `/reports/{report_id}` |  |  | Delete Report |
| GET | `/reports/{report_id}/download` |  |  | Download Report |

## Health, metrics and system status

| Method | Path | Request body | Query parameters | Description |
|---|---|---|---|---|
| GET | `/health/ready` |  |  | Readiness: database reachable and storage directories writable. |
| GET | `/metrics` |  |  | Operational Metrics |
| GET | `/system/status` |  |  | System Status |

## Root

| Method | Path | Request body | Query parameters | Description |
|---|---|---|---|---|
| GET | `/` |  |  | Root |
| GET | `/health` |  |  | Health Check |
