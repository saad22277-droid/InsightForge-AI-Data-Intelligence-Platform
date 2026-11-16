# Windows PowerShell running guide

Every command below was written for **Windows PowerShell** and the real folder layout of this
project. Run backend commands from the **project root** (the folder containing `alembic.ini`),
because the application reads `.env` from there.

> Verification note: the backend, migrations, tests and frontend were verified on Linux
> (Python 3.12, Node 22, PostgreSQL 16). The commands below are standard Windows equivalents;
> if one behaves differently on your machine see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## 1. Prerequisites

| Tool | Version | Check command |
|---|---|---|
| Python | 3.12 or 3.13 (project targets 3.13) | `python --version` |
| Node.js | 20.19+ or 22.12+ (required by Vite 8) | `node --version` |
| PostgreSQL | 14+ (tested with 16) | `psql --version` |
| Git | any recent | `git --version` |

Install with winget if missing:

```powershell
winget install Python.Python.3.13
winget install OpenJS.NodeJS.LTS
winget install PostgreSQL.PostgreSQL.16
```

Close and reopen PowerShell after installing so `PATH` refreshes. If `psql` is not found, use its
full path, e.g. `& "C:\Program Files\PostgreSQL\16\bin\psql.exe" --version`.

## 2. Virtual environment

```powershell
cd path\to\ai-data-platform
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the script ("running scripts is disabled"):

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip check
```

`requirements-dev.txt` = `requirements.txt` (runtime) + `pypdf` (used only by tests). Expect roughly
one to two minutes on first install (`ydata-profiling`, scikit-learn and matplotlib are large).

## 4. Environment file

You already have a `.env` with `DATABASE_URL`. **Keep it** and add the new optional lines from
`.env.example`. To start from scratch instead:

```powershell
Copy-Item .env.example .env
notepad .env
```

Minimum content:

```env
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/ai_data_platform
```

**Gemini key (optional).** The app reads `GEMINI_API_KEY` from the process environment *or* from `.env`.
If you already defined it as a Windows environment variable, nothing else is needed (open a new
terminal after setting it). To check without printing the key:

```powershell
if ($env:GEMINI_API_KEY) { "GEMINI_API_KEY is set" } else { "GEMINI_API_KEY is NOT set" }
```

To put it in `.env` instead, open the file in Notepad and add a line `GEMINI_API_KEY=...`
(never paste the key into chat, commit it, or add it to `.env.example`). Optional:
`GEMINI_MODEL=gemini-2.5-flash` (change if your key cannot use that model).

## 5. Create the database (skip if `ai_data_platform` already exists)

```powershell
psql -U postgres -c "CREATE DATABASE ai_data_platform;"
psql -U postgres -l
```

## 6. Run migrations

```powershell
python -m alembic upgrade head
python -m alembic current
```

* **Fresh database:** creates all tables (`datasets`, `dataset_versions`, `cleaning_runs`, `ml_runs`,
  `chat_conversations`, `chat_messages`, `reports`). Expected `current`: `a1b2c3d4e5f6 (head)`.
* **Existing Stage 4 database:** applies only the new Stage 5 migration; your data is untouched.

## 7. Start the backend

```powershell
python -m uvicorn src.app.main:app --reload --port 8000
```

The first start takes about 20 seconds (large imports). Open http://127.0.0.1:8000/docs for the API
explorer, or check health from a second terminal:

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe http://127.0.0.1:8000/health/ready
curl.exe http://127.0.0.1:8000/ai/status
```

## 8. Install and start the frontend (second PowerShell window)

```powershell
cd path\to\ai-data-platform\frontend
npm install
npm run dev
```

## 9. Open the application

http://localhost:5173 - click **Upload dataset** and choose `data\samples\sales_demo.csv`
(created by `python scripts\generate_sample_data.py`; it deliberately contains missing values,
duplicates, outliers and invalid values so every feature has something to show).

## 10. Tests and verification

```powershell
# Backend (from project root, venv active) - 294 tests
python -m pytest

# Include the real-PostgreSQL migration test (use a scratch database, it is wiped!)
psql -U postgres -c "CREATE DATABASE aidp_migration_test;"
$env:TEST_POSTGRES_URL = "postgresql://postgres:YOUR_PASSWORD@localhost:5432/aidp_migration_test"
python -m pytest tests/test_migrations.py
Remove-Item Env:TEST_POSTGRES_URL

# Frontend (from the frontend folder)
npm run lint
npm run build

# End-to-end smoke test against the running backend (second terminal, venv active)
python scripts\smoke_test.py --skip-ai          # no Gemini quota used
python scripts\smoke_test.py                    # also exercises live Gemini if a key is set
```

## 11. Stop the application

Press `Ctrl + C` in the backend window and in the frontend window. PostgreSQL keeps running as a
Windows service (to stop it, from an Administrator PowerShell: `Stop-Service postgresql-x64-16`).

## 12. Restart later

```powershell
# window 1
cd path\to\ai-data-platform
.\.venv\Scripts\Activate.ps1
python -m uvicorn src.app.main:app --reload --port 8000

# window 2
cd path\to\ai-data-platform\frontend
npm run dev
```

## 13. Generate a report

In the UI: **Reports -> Generate PDF report** (optionally tick *Include AI-written explanations*).
Files are stored in `data\reports\`. Via API:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/datasets/1/reports `
  -ContentType "application/json" -Body '{"include_ai": false}'
```

## 14. Common troubleshooting commands

```powershell
# What is using port 8000 / 5173? Then stop it.
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object LocalPort, OwningProcess
Stop-Process -Id <PID> -Force

# PostgreSQL service and connectivity
Get-Service *postgres*
psql -U postgres -c "SELECT 1;"

# Alembic state
python -m alembic current
python -m alembic history

# Clean reinstall of frontend packages
cd frontend; Remove-Item -Recurse -Force node_modules; npm ci

# Verify package set
python -m pip check
python -m pip list | Select-String "fastapi|pandas|scikit|google-genai|reportlab"
```
