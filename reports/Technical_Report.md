# AI Data Intelligence Platform - Technical Report

**Stage 5: LLM integration, grounded dataset chat, data cleaning, ML studio, experiment tracking and reporting**

Generated from the project repository. All figures in sections 13-15 were measured on the delivered code
(Linux, Python 3.12, Node 22, PostgreSQL 16); nothing in this report is estimated or simulated.

## 1. Project objectives

Turn the Stage 1-4 data platform (upload, profiling, API, dashboard) into a complete, locally runnable AI engineering project that:

- manages datasets and their versions safely;
- cleans data through reviewed, reversible operations;
- performs automated EDA and interactive visualisation;
- trains, evaluates, tracks and serves scikit-learn models without data leakage;
- lets users converse with a dataset through Google Gemini **without the model inventing statistics**;
- produces downloadable analytical reports;
- remains free to run and fully functional without Gemini.

Out of scope by decision: Docker, Kubernetes, Terraform, CI/CD, paid infrastructure (reserved for a separate project).

## 2. What was delivered

| Area | Delivered |
|---|---|
| Backend | 8 routers, 52 API operations, 7 service packages, 5 new tables, 1 new migration (+ fixed baseline) |
| AI | Gemini client with retries and error classes, 10 validated tools, orchestrator, grounding verifier, explanations, persisted chat |
| ML | 8 algorithms across 3 tasks, leakage guards, registry, versioning, comparison, reproduction, prediction |
| Reporting | Multi-page PDF with charts built from real results; CSV column profile |
| Frontend | 11 navigation items; 8 new pages; reusable UI/chart components |
| Quality | 294 automated tests (91% line coverage), 29-step end-to-end smoke test, strict TypeScript, clean lint |

## 3. Architecture

![System architecture](../docs/diagrams/system_architecture.png)

The system is layered: React UI -> FastAPI routers (validation, HTTP mapping) -> service layer (all logic) ->
PostgreSQL + filesystem, with Gemini as an optional external dependency reached only through an `LLMClient`
interface. Routers contain no business logic; services contain no HTTP. This made it possible to test AI
behaviour with a scripted fake model and to keep every non-AI feature independent of Gemini.

## 4. Technology choices

| Choice | Reason |
|---|---|
| FastAPI + Pydantic | Already used; automatic OpenAPI; Pydantic discriminated unions give a closed, validated set of cleaning operations. |
| PostgreSQL + Alembic | Required; JSONB stores flexible run/evidence metadata; migrations make schema changes reproducible. |
| pandas / NumPy / scikit-learn / joblib | Free, standard, deterministic; scikit-learn `Pipeline` + `ColumnTransformer` make leakage prevention structural. |
| Google Gemini via `google-genai` | Required provider; native function calling; free tier. |
| No LangChain / LlamaIndex | One loop and ten tools do not justify a framework; fewer dependencies and fully inspectable prompts. |
| No vector database | Data is tabular: aggregates cannot be retrieved by similarity. A schema card + computational tools is more accurate. |
| No MLflow | A PostgreSQL registry + joblib files meet tracking, versioning and reproducibility needs with no extra service or dependency tree. |
| ReportLab + matplotlib | Pure-Python PDF with real charts; no system binaries (wkhtmltopdf etc.), friendly to Windows. |
| React + TypeScript + Recharts | Already used; strict typing across the API contract; Recharts for standard charts, hand-written SVG for box plots. |

## 5. Database design

Seven tables (`datasets`, `dataset_versions`, `cleaning_runs`, `ml_runs`, `chat_conversations`, `chat_messages`, `reports`);
foreign keys cascade on delete so removing a dataset removes all derived records; JSON columns are JSONB on PostgreSQL.
The original initial migration was empty (the `datasets` table had been created outside Alembic), which made
`alembic upgrade head` fail on a fresh database. It now creates `datasets` only when missing; a Stage 5 migration adds the new
tables. Both a fresh database and a database mimicking the existing state were migrated and verified, and
`alembic check` reports no drift between models and migrations.

## 6. Dataset processing pipeline

![Data and ML pipeline](../docs/diagrams/data_ml_pipeline.png)

1. **Upload**: type/size checks, UUID file name, version 1 recorded with row/column counts; SHA-256 available from metadata.
2. **Analysis**: `data_io.load_dataframe` caches by path+mtime+size and enforces row/column limits.
3. **Cleaning**: `analyze_cleaning` produces typed recommendations (each carries an executable operation); the user queues,
   previews (before/after counts, dtype and statistic changes, sample rows) and applies. Apply writes a *new CSV version*, a
   `DatasetVersion` and a `CleaningRun` audit row; older versions are never modified and can be re-activated.
4. **ML**: the active version feeds training; its number and hash are stored with each run.

## 7. AI architecture and Gemini integration

![Chat flow](../docs/diagrams/chat_flow.png)

**Retrieval approach.** Classic RAG (embed chunks -> vector search -> paste into the prompt) cannot compute an average over
10,000 rows. The platform therefore uses *structured retrieval*: a schema-only dataset card in the prompt, and ten read-only analytical tools
that retrieve exactly the aggregate the question needs. No embeddings or vector store are used.

**Flow.** Gemini receives rules + schema card + conversation history and may request tools. The application validates each
call (schema, column names, ranges, operator enums), executes it with pandas, sanitises and labels the result as untrusted
data, and returns it. Automatic function calling is disabled so the application owns every step; at most six tool executions
are allowed per question.

**Grounding.** After the model writes its answer, every number is compared with the numbers in the tool results (precision-aware
tolerance, percent and k/M/B aware). A failed check triggers one correction request; a second failure replaces the prose with the
raw computed results and a notice. Each message stores its evidence and a verification badge; the UI shows both.

**Reliability.** Timeouts, exponential backoff with jitter for 429/5xx/timeouts, immediate failure for auth/model errors,
classified errors surfaced as structured HTTP errors with `Retry-After`, a per-minute request limiter and concurrency cap that protect the free quota.

**Other AI features.** Explanations of dataset, quality, EDA, correlations, charts and model results share the same facts+verifier
design; PDF reports can embed up to five verified explanations and degrade gracefully if Gemini is unavailable.

## 8. Machine-learning architecture

Pipeline: `ColumnTransformer` (median-impute + scale numeric; mode-impute + one-hot categorical) -> optional `SelectKBest` -> estimator.
Regression (linear, ridge, random forest, gradient boosting), classification (logistic regression, random forest, gradient boosting),
clustering (k-means, k chosen by silhouette). Metrics are computed on a held-out test set and shown with naive baselines and
cross-validation.

**Leakage prevention:** target never a feature; split before any fitting (verified by test: the scaler's `n_samples_seen_` equals
the training rows); automatic exclusion of identifiers, constants, dates, very-high-cardinality text and near-copies of the target;
feature-selection `k` derived from training rows; stratified split.

## 9. Experiment tracking and MLOps

Every run stores configuration (seed, split, hyper-parameters), features, excluded features, metrics, warnings, duration, library
versions, dataset version and SHA-256, plus the artefact path and SHA-256. Capabilities: automatic model versioning, side-by-side
comparison with best-per-metric highlighting, one-click **reproduce** (retrains on the recorded data and reports whether all metrics are
identical), checksum-verified model loading, and a prediction API/UI driven by a stored feature schema.

## 10. API structure

52 operations in 8 routers (full table in `docs/API.md`, generated from OpenAPI): datasets (unchanged), analysis (EDA, charts, metadata,
exports), cleaning (recommend/preview/apply/history/activate), machine learning (catalog/train/compare/runs/predict/reproduce),
experiments (list/summary/compare), AI (status/chat/conversations/explain), reports, system (`/health/ready`, `/metrics`, `/system/status`).
Original contracts, including `/` and `/health`, are preserved.

## 11. Frontend structure

`App.tsx` remains the shell and keeps the original Overview, Datasets, Quality and Versions pages; eight pages were added
(`CleaningPage`, `EdaPage`, `VisualizationsPage`, `MlPage`, `AskAiPage`, `ReportsPage`, `ExperimentsPage`, `SettingsPage`) over shared
components (`ui.tsx`, `charts.tsx`, `AiExplain.tsx`) and a typed API client with centralised error mapping. A `useAsync` hook handles
loading/error/reload state; every page has loading, empty and error states and success notifications.

![Cleaning workflow](fig_cleaning.png)

![ML model detail](fig_ml.png)

## 12. Security controls

Secrets only from the environment (verified: a planted key never appears in any status response); upload size/type limits; closed Pydantic
operation set; no code/SQL/shell execution anywhere; untrusted-data sanitisation and labelling for LLM inputs; checksum- and path-verified
model loading; path-checked report downloads; request-per-minute and concurrency limits; logs exclude dataset contents and chat text
(tested); generic 500 responses. **Not implemented:** authentication (see Limitations).

## 13. Testing methodology

- **Unit tests** for every deterministic component (EDA, cleaning, charts, ML, tools, grounding) compared against pandas results.
- **Integration tests** run the real routers with SQLite and a scripted `FakeLLM`; Gemini is never contacted.
- **Provider tests** replace the SDK client to test retry/backoff, error mapping, empty/blocked responses and missing keys.
- **Adversarial tests**: oversized/odd uploads, path traversal names, tampered model file, model path outside storage, report path escape,
  unknown tools (`run_python`), prompt-injection strings in cells, hallucinated numbers, quota/rate-limit errors.
- **Migration test** against real PostgreSQL (fresh schema) plus model/migration drift check.
- **End-to-end smoke test** against a running server and PostgreSQL (29 checks).
- Frontend: strict TypeScript build and ESLint; manual visual verification in headless Chromium.

## 14. Actual results

| Check | Result |
|---|---|
| Full backend suite (fresh virtual environment from `requirements-dev.txt`) | **294 passed** in 41 s (with PostgreSQL migration test) |
| Of which original Stage 1-4 tests | 111 passed, unmodified |
| Line coverage of `src/app` | **91%** (3,906 statements, 362 missed) |
| Fresh-database migration | `upgrade head` succeeded; `alembic check`: no drift |
| Existing-database upgrade | succeeded, data preserved |
| `pip check` | No broken requirements |
| End-to-end smoke test | **29 / 29** passed (about 24 s) |
| Frontend lint / build | clean / succeeded (`tsc -b` + Vite) |
| Server log during E2E | 0 ERROR-level lines |

Models trained by the smoke test on the demo dataset (812 rows, after cleaning): regression of `Profit` with random forest,
R² 0.739 against a baseline R² of -0.022; classification of `Category` with gradient boosting, accuracy 0.894 against a
majority baseline; k-means with k = 2 chosen automatically, silhouette 0.244. Reproducing a run yielded identical metrics.
These numbers illustrate behaviour on synthetic demo data; they are not benchmark claims.

Defects found and fixed by this process (selected): fresh-database migration failure; missing `setuptools<81` pin; a logging middleware
that swallowed exceptions the original tests expect; a validator that never fired when its field was omitted; chat `updated_at` set before
being populated; a grounding tolerance that accepted "2019" for "2023"; a stale `last_error` in the AI status endpoint; classification
leakage via a numeric column derived from the target; and feature-selection `k` derived using test rows.

## 15. Performance observations

Measured with the running server on a shared sandbox CPU, 812 rows x 11 columns (median of 2-5 calls):

| Operation | Median |
|---|---:|
| EDA (`GET /datasets/{id}/eda`) | 117 ms |
| Chart (histogram) | 8 ms |
| Cleaning recommendations | 63 ms |
| Train linear regression | 244 ms |
| Train random forest (100 trees, CV 5) | 2.6 s |
| Train gradient-boosting classifier (CV 5) | 4.6 s |
| Train k-means with automatic k | 420 ms |
| Predict (1 record) | 25 ms |
| Generate PDF report (no AI) | 2.0 s |

Backend start-up takes about 20 s (large imports, notably ydata-profiling). Gemini latency was **not measured** (no key available).
Training is synchronous; larger datasets will take proportionally longer (rows are capped at 100,000 for training).

## 16. Limitations

Live Gemini was not exercised; not run on Windows or Python 3.13; no authentication; synchronous training; CSV only; heuristics-based
cleaning recommendations; single-process metrics and rate limiting. See `docs/LIMITATIONS.md` for the complete list.

## 17. Future improvements

1. Authentication, per-user workspaces and persistent rate limiting.
2. Background job queue for training and reports with progress reporting.
3. Streaming Gemini responses (SSE) and chart-aware chat answers.
4. Hyper-parameter search, SHAP explanations, time-series support and feature engineering for dates.
5. Export of runs to MLflow if a shared tracking server becomes useful.
6. Additional import formats (Excel, Parquet, database connections).
7. Automated browser tests (Playwright) and a CI pipeline in the follow-up infrastructure project.
8. Optional pgvector index for dataset documentation / data dictionaries.
