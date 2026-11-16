# Security controls

| Area | Control | Where |
|---|---|---|
| Secrets | `GEMINI_API_KEY` and `DATABASE_URL` come from the environment / `.env`; never hard-coded, never returned by any endpoint (a test greps every status endpoint for a planted key), never logged. `.env` is git-ignored; `.env.example` has placeholders only. | `config.py`, `.gitignore`, `tests/test_api_system.py` |
| Uploads | Extension and content checks (existing), **size limit** (`MAX_UPLOAD_MB`, HTTP 413), row/column limits on load, UUID-prefixed server-side file names so client names cannot traverse paths. | `routes/datasets.py`, `services/data_io.py` |
| Input validation | Pydantic models for every body (bounded lists, enums, ranges); cleaning operations are a closed discriminated union - unknown operations are 422. Query parameters have length/range constraints. | `schemas/*`, `routes/*` |
| Code execution | None. The LLM cannot execute Python/SQL/shell; cleaning operations and ML algorithms are fixed whitelists. | `services/ai/tools.py`, `services/cleaning.py` |
| Prompt injection | Untrusted-data labelling + sanitisation + tool whitelist + grounding verifier (see [AI_ARCHITECTURE.md](AI_ARCHITECTURE.md)). | `services/ai/*` |
| Model files | Loaded only from the model directory, only if the SHA-256 matches the registry (tamper test). | `services/ml/service.py` |
| Report downloads | Path must resolve inside the reports directory. | `routes/reports.py` |
| Abuse / cost | AI request-per-minute limiter, concurrency cap, tool-call budget, row caps, message-length cap. | `services/ai/guard.py`, `config.py` |
| Logging | Structured JSON; request-id, route template, status, duration. Dataset values, questions and answers are never logged (tested). | `main.py`, `logging_config.py` |
| Error handling | Central handler for AI errors; no stack traces or file paths in responses; unexpected exceptions return a generic 500. | `main.py` |
| CORS | Origins from `CORS_ORIGINS` (defaults to the local Vite dev server). | `main.py` |
| Rendering | LLM output is rendered as React text nodes (no `dangerouslySetInnerHTML`); PDF text is XML-escaped. | `frontend/src/components/ui.tsx`, `services/reporting/builder.py` |

## Known gaps (be aware before deploying anywhere public)

* **There is no authentication or authorisation** (the original project had none). Anyone who can reach the API can read every
  dataset, run models and spend your Gemini quota. Run it locally or add auth before exposing it.
* The upload response of the original `/datasets/upload` still includes the server-side `file_path`.
* Rate limiting is per-process (in memory).
* joblib/pickle artefacts are safe only because the app writes and checksum-verifies them itself.
* Free-tier Gemini data-use terms: do not send confidential data.
