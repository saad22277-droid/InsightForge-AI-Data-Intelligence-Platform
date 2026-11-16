# Project limitations (honest list)

**Verification gaps**
* **Live Gemini was not exercised.** The build environment had no API key. All Gemini behaviour is verified
  against a mocked SDK client and a scripted fake model (retries, error classes, tool loop, grounding, fallbacks).
  Use *Settings -> Test Gemini connection* or `python scripts/smoke_test.py` on your machine.
* Verified on Linux with Python 3.12, Node 22 and PostgreSQL 16. Not run on Windows or Python 3.13; the dependency pins are
  those that installed cleanly (`pip check` clean) in a fresh environment.
* Default model `gemini-2.5-flash` may be retired or renamed by Google; change `GEMINI_MODEL` (the app reports `model_unavailable`).
* The screenshot of "Ask AI" shows the unconfigured state; no real answer was captured.

**Functional**
* CSV only (no Excel/Parquet/database import). Datasets are fully loaded in memory (default limits: 25 MB upload, 500k rows, 500 columns).
* Training runs synchronously inside the request (seconds for the demo data; no job queue). Large grids/slow models can hit client timeouts.
* Sampling to 100k rows for training; no time-series models, no hyper-parameter search, no deep learning, no SHAP.
* Clustering is k-means only; metrics are internal (no ground truth).
* Datetime columns are excluded from ML features rather than engineered.
* Cleaning is single-table and rule-based; recommendations are heuristics (e.g. "non-negative" detection uses column-name hints).
* Dataset chat answers questions that the 10 tools can compute; it cannot join datasets, forecast, or run custom code.
  Number grounding checks *numbers*, not the correctness of qualitative interpretation.
* The original ydata-profiling endpoint (`/datasets/{id}/profile`) is unchanged and can be slow on large data.
* The tracker is not MLflow-compatible out of the box; there is no model-serving process separate from the API.

**Operational**
* No authentication, multi-user separation, or persistent rate limiting (see [SECURITY.md](SECURITY.md)).
* Metrics are in-memory and reset on restart.
* No Docker/Kubernetes/CI by design (reserved for a separate project).
