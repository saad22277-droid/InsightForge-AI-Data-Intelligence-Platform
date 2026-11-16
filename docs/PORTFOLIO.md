# Portfolio presentation

## GitHub repository description (one line)

Local-first AI data platform: clean, explore and model datasets, track ML experiments, and chat with your data through Google Gemini - with every number verified against computed results. FastAPI - PostgreSQL - scikit-learn - React/TypeScript.

**Suggested topics:** `fastapi` `react` `typescript` `postgresql` `scikit-learn` `google-gemini` `llm` `tool-calling` `mlops` `data-science` `eda` `data-cleaning`

## Polished overview

AI Data Intelligence Platform is a full-stack application that takes a CSV from upload to insight: it recommends and applies reviewed
data-cleaning steps (never touching the original file), produces automated exploratory analysis and interactive charts, trains and tracks
scikit-learn models with built-in leakage protection, and lets users interrogate a dataset in natural language.

The AI layer is designed around a hard problem in LLM engineering - **hallucinated numbers**. Gemini never calculates: it selects from ten
validated, read-only analysis tools, deterministic pandas code produces the figures, and a verifier rejects any answer containing a number
that was not computed. The UI exposes the evidence for every answer. If Gemini is unavailable, everything else keeps working.

## Feature highlights

* Cleaning engine with recommendations, queue, before/after preview, versioned apply and rollback
* Automated EDA, quality score, 8 interactive chart types (incl. SVG box plots and correlation heat-map)
* ML studio: 8 algorithms, train/test split, CV, baselines, importance, confusion matrix, PCA clusters, prediction form
* Experiment tracker: model versions, data fingerprints, comparison, one-click reproduction, checksum-verified artefacts
* Grounded dataset chat with persisted conversations, tool-call evidence and verification badges
* One-click PDF analytical reports with charts, model results and optional verified AI commentary
* Ops: structured logs, request IDs, health/ready/metrics endpoints, AI status/quota guard
* 294 automated tests (91% coverage), 29-step end-to-end smoke test, strict TypeScript

## Technology stack

Python - FastAPI - Pydantic - SQLAlchemy - Alembic - PostgreSQL - pandas - NumPy - scikit-learn - joblib - ReportLab - matplotlib -
Google Gemini (`google-genai`) - React 19 - TypeScript - Vite - Recharts - pytest.

## Architecture summary

React UI -> FastAPI routers (validation only) -> service layer (all logic) -> PostgreSQL + filesystem. The AI path is
question -> Gemini plans tool calls -> argument validation -> pandas tools -> sanitised results -> Gemini explains -> grounding verifier ->
persisted answer + evidence. See `docs/diagrams/`.

## Engineering stories worth telling in an interview

* **Hallucination control**: designed a precision-aware number verifier; found and fixed a tolerance bug that would have accepted "2019" for "2023".
* **Leakage by construction**: pipelines fit only on training rows; tests assert it; derived-target columns are detected and excluded.
* **Safe by architecture**: no code/SQL execution, tool whitelist, sanitised untrusted data, checksum-verified model loading.
* **Evolving a legacy codebase**: kept 111 original tests green, fixed a latent fresh-install migration failure without rewriting history.
* **Honest verification**: mocked SDK for CI-style determinism, plus a smoke test that exercises the live stack and, with a key, the live API.

## Screenshots to capture for the repository / portfolio

Already in `docs/screenshots/`: overview, datasets, cleaning preview, EDA correlations, box plot, ML models, reports, experiments, settings.
**Capture yourself with your Gemini key:** (1) Ask AI after answering "Which columns have the most missing values?" with the *How this was computed*
panel expanded, (2) Ask AI showing the green "Numbers verified" badge on a multi-turn conversation, (3) ML Studio -> Explain these results,
(4) Settings -> Test Gemini connection showing "Working", (5) the PDF report with Gemini sections (open `data/reports/*.pdf`).

## Resume / portfolio blurb

> **AI Data Intelligence Platform** - Built a full-stack AI analytics platform (FastAPI, PostgreSQL, scikit-learn, React/TypeScript, Google Gemini). Designed a grounded LLM
> architecture using validated read-only tool calling and a numeric-verification layer that blocks hallucinated statistics; implemented a versioned data-cleaning engine,
> leakage-safe ML pipelines with experiment tracking and reproducibility, and automated PDF reporting. 294 automated tests (91% coverage), fresh-database migrations,
> structured logging and health/metrics endpoints.
