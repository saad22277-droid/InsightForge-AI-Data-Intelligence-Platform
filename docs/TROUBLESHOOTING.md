# Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `RuntimeError: DATABASE_URL is not set` / backend exits at start | `.env` missing or you are not in the project root | Run from the folder containing `.env`; see `.env.example`. |
| `psycopg2.OperationalError: connection refused` / `password authentication failed` | PostgreSQL stopped, wrong port or password | `Get-Service *postgres*`; start it; fix `DATABASE_URL`. |
| `alembic upgrade head` fails with `relation "datasets" does not exist` | You are on an old copy of the initial migration | Use the Stage 5 version of `be1adc90fcf0_initial_database_schema.py` (creates `datasets` when missing). |
| `ModuleNotFoundError: pkg_resources` | `setuptools` >= 81 installed | `python -m pip install "setuptools<81"` (pinned in `requirements.txt`). |
| UI shows "Cannot reach the backend at http://127.0.0.1:8000" | Backend not running or different port | Start the backend; if you use another port set `VITE_API_BASE_URL` in `frontend/.env` and add the origin to `CORS_ORIGINS`. |
| Ask AI shows "Gemini is not configured" | `GEMINI_API_KEY` missing | Set it in `.env` or as a Windows environment variable, restart the backend and open a new terminal. |
| AI error `auth` (HTTP 502) | Key rejected | Create a new key at https://aistudio.google.com/apikey and update `.env`. |
| AI error `model_unavailable` | `GEMINI_MODEL` retired or not available to your key | Set `GEMINI_MODEL` to a model listed in Google AI Studio and restart. |
| AI error `quota` / `rate_limited` (HTTP 429) | Free-tier quota or the app's own limiter (`AI_REQUESTS_PER_MINUTE`) | Wait for the `Retry-After` period; lower usage; check your quota in AI Studio. Non-AI features are unaffected. |
| AI error `timeout` (HTTP 504) | Slow network or large request | Increase `GEMINI_TIMEOUT_SECONDS`; retry. |
| Assistant shows "Showing raw computed results" | The model kept writing numbers that are not in the computed results; the grounding verifier replaced the text | Expected safety behaviour. Rephrase the question or retry. |
| `413 File is too large` on upload | Above `MAX_UPLOAD_MB` (default 25) | Raise `MAX_UPLOAD_MB` or sample the file. |
| `400 Dataset has N rows; the limit is ...` | Above `MAX_DATASET_ROWS` / `MAX_DATASET_COLUMNS` | Raise the limits in `.env`. |
| Model training returns 400 | Invalid configuration (text target for regression, class with one row, target used as feature ...) | The message states exactly what to change. |
| Predict returns 409 "checksum mismatch" | The `.joblib` file was modified or replaced | Retrain; models are only loaded when their SHA-256 matches the registry. |
| Report is missing AI text | Gemini unavailable or `Include AI` unchecked | The PDF notes the reason (section "AI-generated explanations"). |
| `npm run dev` fails with a Node version error | Node older than 20.19 / 22.12 | Install the current Node LTS. |
| PowerShell: "running scripts is disabled" | Execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Backend start takes ~20 s | `ydata-profiling` import | Normal. |
| Windows line-ending warnings (`LF will be replaced by CRLF`) | Git autocrlf | Harmless; see `git config core.autocrlf`. |

Useful endpoints while debugging: `/health/ready`, `/ai/status?check=true`, `/system/status`, `/metrics`.
Logs are structured JSON on stdout (`LOG_JSON=false` for plain text). Dataset contents and chat text
are never logged.
