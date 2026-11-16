# Machine learning architecture

Implemented with scikit-learn (`services/ml/training.py`, registry in `services/ml/service.py`).
Nothing is simulated: every metric on screen is computed on a held-out test set from the real data.

## Tasks and algorithms

| Task | Algorithms | Primary metric | Other metrics |
|---|---|---|---|
| Regression | linear regression, ridge, random forest, gradient boosting | R² | MAE, RMSE, baseline R²/MAE (predict-the-mean), k-fold CV R² on training rows, residual stats, predicted-vs-actual sample |
| Classification | logistic regression, random forest, gradient boosting | weighted F1 | accuracy, balanced accuracy, weighted precision/recall, macro F1, ROC-AUC (binary and one-vs-rest), baseline accuracy (majority class), confusion matrix, per-class report, CV F1 |
| Clustering | k-means (k chosen automatically by silhouette over k=2..8, or user-specified) | silhouette | Davies-Bouldin, Calinski-Harabasz, inertia, cluster sizes and profiles, 2-D PCA projection |

Hyper-parameters are whitelisted per algorithm and range-checked (`n_estimators`, `max_depth`,
`learning_rate`, `alpha`, `C`). Feature selection: optional `SelectKBest` *inside* the pipeline.

## Pipeline

```text
ColumnTransformer
  numeric     -> SimpleImputer(median) -> StandardScaler
  categorical -> SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown="ignore")
[SelectKBest(k)]  (optional)
Estimator
```

## Preventing data leakage

1. The target can never be a feature (explicit selection is rejected with a clear message).
2. The train/test split happens **before** anything is fitted; imputers, scalers, encoders and feature
   selection are fitted on the training partition only (a test asserts the scaler saw only training rows).
   `k` for SelectKBest is resolved from the training partition.
3. Automatic exclusions, reported to the user: identifier-like columns (>=95% unique), constant/empty columns,
   date columns, text with more than 50 levels, near-duplicates of the target (regression: |Pearson| or |Spearman| >= 0.999;
   classification: a categorical column that maps one-to-one onto the target, or a numeric column a depth-3 tree can
   classify with >=99.9% accuracy).
4. Stratified split for classification; every class needs at least 2 rows.
5. Cross-validation is run on the training partition only.
6. Metrics are always shown next to a naive baseline so "good-looking" numbers can be put in context, and an
   overfitting warning is raised when train performance exceeds test by more than 0.2.

## Invalid-input handling

Non-numeric regression targets, <5 distinct regression target values, single-class targets, >50 classes, continuous
float targets for classification, fewer than `ML_MIN_ROWS` (20) usable rows, unsplittable data, unknown columns,
wrong algorithm for the task and unsupported hyper-parameters all fail with HTTP 400 and a specific message; nothing is
recorded for invalid requests. Large datasets are sampled to `ML_MAX_TRAIN_ROWS` with a visible warning.

## Model registry, experiment tracking and reproducibility (MLOps)

Each run is one `ml_runs` row (see [DATABASE.md](DATABASE.md)):

* **Versioning** - `(dataset, model_name)` -> auto-incrementing `version`.
* **Metadata** - configuration (seed, split, hyper-parameters), features used and excluded, metrics, primary metric,
  warnings, duration, environment (Python / scikit-learn / pandas / NumPy versions).
* **Artefacts** - joblib file in `data/models/<dataset_id>/`, with SHA-256 in the registry. Loading verifies the checksum
  and that the path is inside the model directory (joblib is pickle-based, so only trusted files may be loaded).
* **Data lineage** - dataset version number and file SHA-256 per run.
* **Reproduce** - `POST /ml/runs/{id}/reproduce` retrains with the stored configuration on the *recorded* dataset version,
  refuses if the file hash changed, and reports whether all metrics are identical.
* **Comparison** - train several algorithms at once (`/ml/compare`) or compare any 2-6 runs of the same task
  (`/experiments/compare`); the best value per metric is highlighted (lower-is-better metrics handled).
* **Prediction** - schema endpoint drives the UI form; predictions return probabilities for classifiers and warn about unknown
  or missing fields (missing values use the training imputation).

**Why not MLflow?** It was evaluated and intentionally not added. Its tracking server, artefact store and dependency tree
duplicate what the PostgreSQL registry + joblib files already provide for this single-user, local project, and add
installation risk on Windows. The tracker's data model (run, params, metrics, artefacts, tags) maps directly onto MLflow if you
later want to export runs.
