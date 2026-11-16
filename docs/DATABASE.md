# Database design

PostgreSQL via SQLAlchemy 2.x; schema managed exclusively by Alembic.

## Tables

| Table | Purpose | Key columns |
|---|---|---|
| `datasets` | One row per uploaded dataset (Stage 3) | `id`, `filename`, `file_path` (active version), `profile_path`, `row_count`, `column_count`, `status`, `created_at` |
| `dataset_versions` | Every stored CSV version (Stage 3) | `dataset_id` FK, `version_number` (unique per dataset), `file_path`, counts |
| `cleaning_runs` | Audit trail of applied cleaning | `dataset_id`, `source_version`, `result_version`, `operations` JSONB, `summary` JSONB |
| `ml_runs` | Experiment tracking + model registry | `dataset_id`, `dataset_version_number`, `dataset_sha256`, `model_name`, `version`, `task`, `algorithm`, `target`, `features`, `config`, `metrics`, `primary_metric(_value)`, `details`, `environment`, `artifact_path`, `artifact_sha256`, `duration_ms`; unique `(dataset_id, model_name, version)` |
| `chat_conversations` | Dataset chat threads | `dataset_id` FK, `title`, timestamps |
| `chat_messages` | Messages | `conversation_id` FK, `role`, `content`, `evidence` JSONB (tool calls + results), `grounding` JSONB, `model`, `latency_ms` |
| `reports` | Generated PDF reports | `dataset_id` FK, `title`, `file_path`, `size_bytes`, `include_ai`, `ai_included`, `sections` JSONB |

All new foreign keys use `ON DELETE CASCADE`, so deleting a dataset (or "Clear workspace") removes its
chats, runs, reports and audit rows without changing the existing `Dataset` model. JSON columns use
`JSONB` on PostgreSQL and generic `JSON` on SQLite (used by the test-suite).

```text
datasets 1 ──< dataset_versions
    │ 1 ──< cleaning_runs
    │ 1 ──< ml_runs
    │ 1 ──< reports
    └ 1 ──< chat_conversations 1 ──< chat_messages
```

## Migrations

| Revision | Description |
|---|---|
| `be1adc90fcf0` | Initial schema. **Was empty** (the `datasets` table had been created outside Alembic). It now creates `datasets` *only if it does not exist*, so fresh databases work and existing ones are unaffected. |
| `13bdfc2dfb15` | Adds `dataset_versions` |
| `73db4e02e19b` | Unique `(dataset_id, version_number)` |
| `a1b2c3d4e5f6` | **Stage 5**: `cleaning_runs`, `ml_runs`, `chat_conversations`, `chat_messages`, `reports` |

Verification performed: `alembic upgrade head` on an empty PostgreSQL 16 database; upgrade of a database
that mimics your existing state (tables created pre-Alembic, stamped at `73db4e02e19b`) preserving data;
`alembic check` reports no drift between models and migrations. The same fresh-database test is part of the
test-suite (`tests/test_migrations.py`, enabled by `TEST_POSTGRES_URL`).

## Initialise a fresh database

```powershell
psql -U postgres -c "CREATE DATABASE ai_data_platform;"
python -m alembic upgrade head
```
