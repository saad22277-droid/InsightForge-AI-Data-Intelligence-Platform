# Testing

## How to run

```powershell
python -m pytest                     # 293 pass + 1 skipped (the PostgreSQL migration test, see below)
python -m pytest -q tests/test_orchestrator.py
$env:TEST_POSTGRES_URL = "postgresql://postgres:PASSWORD@localhost:5432/scratch_db"   # WIPES that database
python -m pytest tests/test_migrations.py                                              # 294 pass, 0 skipped
cd frontend; npm run lint; npm run build
python scripts\smoke_test.py --skip-ai   # end-to-end against a running backend (29 checks)
```

Normal automated tests **never** call the live Gemini API and need no API key or PostgreSQL:
API tests run the real FastAPI routers on an in-memory SQLite database with a scripted `FakeLLM`.

## Suite (verified results)

| File | Tests | Covers |
|---|---:|---|
| `test_main.py` (original) | 111 | Stage 1-4 behaviour, unchanged and still passing |
| `test_ml_training.py` | 23 | metrics, all algorithms, leakage guards, validation, reproducibility, feature selection |
| `test_grounding.py` | 18 | number verification (rounding, %, k/M, invented numbers) |
| `test_orchestrator.py` | 16 | tool loop, correction retry, fallback, tool budget, injection, history cap, rate limit |
| `test_gemini_client.py` | 15 | mocked SDK: retries/backoff, error classification, blocked/empty, missing key, thought-signature replay |
| `test_api_ai.py` | 15 | chat persistence, 503/429/504 mapping, explain endpoints, validation |
| `test_charts.py` | 14 | every chart type vs pandas, invalid requests |
| `test_ai_tools.py` | 14 | tool validation, limits, sanitisation, never-raises |
| `test_cleaning.py` | 13 | each operation, immutability of the original, schema rejection, recommendations |
| `test_api_ml.py` | 13 | train/versioning/predict/compare/reproduce/tamper/path-escape/delete |
| `test_eda.py` | 9 | exact statistics, outliers, insights, strict-JSON |
| `test_api_system.py` | 9 | health, metrics, key never leaked, AI status, log hygiene |
| `test_api_data_pipeline.py` | 9 | upload limits, cleaning workflow + rollback, deletion of artefacts |
| `test_api_reports.py` | 8 | PDF content from real data, AI/no-AI/outage, traversal, edge datasets |
| `test_data_io.py` | 4 | caching, limits, JSON safety |
| `test_migrations.py` | 3 | single head, models == migration, fresh PostgreSQL (opt-in) |
| **Total** | **294** | |

Latest run (Linux, Python 3.12, fresh virtual environment from `requirements-dev.txt`):
`294 passed` with `TEST_POSTGRES_URL` set; `293 passed, 1 skipped` without it.
Measured line coverage of `src/app`: **91%** (3,906 statements; lowest: `services/ai/tools.py` 77%, `provider.py` 76%).

## End-to-end smoke test

`scripts/smoke_test.py` drives a real server and PostgreSQL: upload, metadata, EDA, six chart types,
cleaning recommend/preview/apply, regression/classification/clustering training, comparison, prediction,
reproduction, model versioning, experiment tracker and comparison, AI status, PDF generation and download,
version rollback, metrics. Result: **29/29 passed** (about 24 s). With a key, it also runs a live Gemini
connection test, a grounded chat answer and multi-turn persistence.

## Frontend

* `npm run lint` (ESLint 10 + react-hooks 7 + typescript-eslint): clean.
* `npm run build` (`tsc -b` strict type-check + Vite production build): succeeds.
* Pages were exercised in headless Chromium against the running stack (13 screenshots in `docs/screenshots/`);
  no browser console errors were reported. There are no automated browser tests.

## Mocking policy

| Real | Mocked |
|---|---|
| pandas / NumPy / scikit-learn computations, CSV files, joblib files, PDF generation, FastAPI routing, SQLAlchemy models | Gemini (`FakeLLM`, fake SDK client), PostgreSQL -> SQLite in API tests, `ydata-profiling` HTML generation in upload tests |
