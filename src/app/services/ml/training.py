"""Scikit-learn training pipelines (pure functions, no database access).

Leakage protection
------------------
* The target is never part of the features (explicit selection is rejected).
* The train/test split happens **before** any fitted preprocessing; imputation,
  scaling, one-hot encoding and feature selection live inside a single sklearn
  ``Pipeline`` that is fitted on the training partition only.
* Identifier-like columns and near-copies of the target are excluded and reported.
* Metrics are computed on the held-out test set and compared with a naive baseline.
"""
from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.app.config import get_settings
from src.app.services.data_io import clip_text, json_safe, numeric_columns
from src.app.services.eda import detect_datetime_columns

MAX_CATEGORICAL_LEVELS = 50
MAX_CLASSES = 50

ALGORITHMS = {
    "regression": ["linear_regression", "ridge", "random_forest", "gradient_boosting"],
    "classification": ["logistic_regression", "random_forest", "gradient_boosting"],
    "clustering": ["kmeans"],
}
PRIMARY_METRIC = {
    "regression": "r2",
    "classification": "f1_weighted",
    "clustering": "silhouette",
}
# name -> (type, min, max)
HYPERPARAMETER_RULES = {
    "n_estimators": (int, 5, 500),
    "max_depth": (int, 1, 50),
    "learning_rate": (float, 0.001, 1.0),
    "alpha": (float, 0.0001, 1000.0),
    "C": (float, 0.0001, 1000.0),
    "min_samples_leaf": (int, 1, 100),
}
ALLOWED_HYPERPARAMETERS = {
    "ridge": {"alpha"},
    "logistic_regression": {"C"},
    "random_forest": {"n_estimators", "max_depth", "min_samples_leaf"},
    "gradient_boosting": {"n_estimators", "max_depth", "learning_rate"},
    "linear_regression": set(),
    "kmeans": set(),
}


@dataclass
class TrainingResult:
    pipeline: Pipeline
    task: str
    algorithm: str
    target: str | None
    features: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    classes: list[str] | None
    metrics: dict
    details: dict
    warnings: list[str] = field(default_factory=list)
    duration_ms: int = 0
    effective_config: dict = field(default_factory=dict)


# --------------------------------------------------------------------- validation
def _validate_hyperparameters(algorithm: str, hyperparameters: dict) -> dict:
    allowed = ALLOWED_HYPERPARAMETERS.get(algorithm, set())
    clean = {}
    for key, value in hyperparameters.items():
        if key not in allowed:
            raise ValueError(
                f"Hyperparameter '{key}' is not supported for {algorithm}. "
                f"Supported: {sorted(allowed) or 'none'}"
            )
        kind, low, high = HYPERPARAMETER_RULES[key]
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Hyperparameter '{key}' must be a number")
        if not low <= value <= high:
            raise ValueError(f"Hyperparameter '{key}' must be between {low} and {high}")
        clean[key] = kind(value)
    return clean


def _build_model(task: str, algorithm: str, hp: dict, random_state: int):
    if algorithm not in ALGORITHMS[task]:
        raise ValueError(
            f"Algorithm '{algorithm}' is not available for {task}. "
            f"Choose from: {', '.join(ALGORITHMS[task])}"
        )
    if task == "regression":
        return {
            "linear_regression": lambda: LinearRegression(),
            "ridge": lambda: Ridge(random_state=random_state, **hp),
            "random_forest": lambda: RandomForestRegressor(
                n_estimators=hp.pop("n_estimators", 100), random_state=random_state,
                n_jobs=1, **hp),
            "gradient_boosting": lambda: GradientBoostingRegressor(
                random_state=random_state, **hp),
        }[algorithm]()
    if task == "classification":
        return {
            "logistic_regression": lambda: LogisticRegression(max_iter=2000, **hp),
            "random_forest": lambda: RandomForestClassifier(
                n_estimators=hp.pop("n_estimators", 100), random_state=random_state,
                n_jobs=1, **hp),
            "gradient_boosting": lambda: GradientBoostingClassifier(
                random_state=random_state, **hp),
        }[algorithm]()
    return KMeans(n_init=10, random_state=random_state, **hp)


# -------------------------------------------------------------- feature preparation
def select_features(
    df: pd.DataFrame, target: str | None, requested: list[str] | None, task: str
) -> tuple[list[str], list[str], list[str], list[dict]]:
    """Return (numeric_features, categorical_features, all_features, excluded)."""
    if requested:
        for column in requested:
            if column not in df.columns:
                raise ValueError(f"Feature column not found: {column}")
        if target and target in requested:
            raise ValueError(
                f"The target column '{target}' cannot also be a feature (data leakage)."
            )
        candidates = list(dict.fromkeys(requested))
    else:
        candidates = [c for c in df.columns if c != target]

    rows = len(df)
    datetime_cols = set(detect_datetime_columns(df))
    numeric_set = set(numeric_columns(df))
    excluded: list[dict] = []
    kept_numeric, kept_categorical = [], []

    target_series = df[target] if target else None
    for column in candidates:
        series = df[column]
        reason = None
        non_null = series.dropna()
        if non_null.empty:
            reason = "column is empty"
        elif non_null.nunique() <= 1:
            reason = "column is constant"
        elif column in datetime_cols or pd.api.types.is_datetime64_any_dtype(series):
            reason = "date/time column (not used as a raw feature)"
        elif (
            column not in numeric_set
            and non_null.nunique() > MAX_CATEGORICAL_LEVELS
        ):
            reason = f"more than {MAX_CATEGORICAL_LEVELS} distinct text values"
        elif (
            rows >= 20
            and not pd.api.types.is_float_dtype(series)
            and non_null.nunique() / max(len(non_null), 1) >= 0.95
        ):
            reason = "identifier-like (almost every value is unique)"
        elif target_series is not None and _is_target_leak(series, target_series, task):
            reason = "near-duplicate of the target (data leakage)"
        if reason:
            excluded.append({"column": column, "reason": reason})
        elif column in numeric_set or pd.api.types.is_bool_dtype(series):
            kept_numeric.append(column)
        else:
            kept_categorical.append(column)

    features = kept_numeric + kept_categorical
    if not features:
        raise ValueError(
            "No usable feature columns remain after exclusions: "
            + "; ".join(f"{e['column']} ({e['reason']})" for e in excluded[:5])
        )
    return kept_numeric, kept_categorical, features, excluded


def _is_target_leak(feature: pd.Series, target: pd.Series, task: str) -> bool:
    """Heuristics for features that (almost) *are* the target."""
    frame = pd.DataFrame({"f": feature, "t": target}).dropna()
    if len(frame) < 10:
        return False
    numeric_feature = pd.api.types.is_numeric_dtype(frame["f"]) and not (
        pd.api.types.is_bool_dtype(frame["f"]))

    if task == "regression" and numeric_feature:
        t = pd.to_numeric(frame["t"], errors="coerce")
        if t.nunique() > 1 and frame["f"].nunique() > 1:
            f = frame["f"].astype(float)
            pearson = abs(f.corr(t))
            spearman = abs(f.corr(t, method="spearman"))
            return bool(max(pearson, spearman) >= 0.999)
        return False

    if task == "classification":
        classes = frame["t"].nunique()
        if classes < 2:
            return False
        levels = frame["f"].nunique()
        if not numeric_feature or levels <= 2 * classes:
            # categorical-like feature that maps (almost) one-to-one onto the target
            if 2 <= levels <= max(2 * classes, 2):
                return bool(frame.groupby("f")["t"].nunique().max() == 1)
        if numeric_feature and len(frame) >= 50:
            # a single numeric column that separates the classes (almost) perfectly
            tree = DecisionTreeClassifier(max_depth=3, random_state=0)
            tree.fit(frame[["f"]].astype(float), frame["t"])
            return bool(tree.score(frame[["f"]].astype(float), frame["t"]) >= 0.999)
    return False


def coerce_features(
    df: pd.DataFrame, numeric: list[str], categorical: list[str]
) -> pd.DataFrame:
    """Shared by training and prediction so the two cannot drift apart."""
    data = {}
    for column in numeric:
        series = df[column] if column in df.columns else pd.Series(np.nan, index=df.index)
        data[column] = pd.to_numeric(series, errors="coerce")
    for column in categorical:
        series = df[column] if column in df.columns else pd.Series(np.nan, index=df.index)
        data[column] = series.where(series.isna(), series.astype(str)).astype(object)
    return pd.DataFrame(data, index=df.index)[numeric + categorical]


def _preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric:
        transformers.append((
            "num",
            Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "cat",
            Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False))]),
            categorical,
        ))
    return ColumnTransformer(transformers, remainder="drop")


def _feature_schema(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> list[dict]:
    schema = []
    for column in numeric:
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        schema.append({"name": column, "kind": "numeric",
                       "min": values.min(), "max": values.max(),
                       "mean": values.mean(), "median": values.median()})
    for column in categorical:
        counts = df[column].dropna().astype(str).value_counts()
        schema.append({"name": column, "kind": "categorical",
                       "values": [clip_text(v, 60) for v in counts.head(30).index]})
    return json_safe(schema)


def _importance(pipeline: Pipeline, numeric: list[str], categorical: list[str]) -> list[dict]:
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        raw = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        raw = np.abs(np.asarray(model.coef_, dtype=float))
        raw = raw.mean(axis=0) if raw.ndim > 1 else raw
    else:
        return []
    names = list(pipeline.named_steps["prep"].get_feature_names_out())
    if "select" in pipeline.named_steps:
        mask = pipeline.named_steps["select"].get_support()
        names = [n for n, keep in zip(names, mask) if keep]
    if len(names) != len(raw):
        return []
    totals: dict[str, float] = {}
    for name, value in zip(names, raw):
        origin = name
        if name.startswith("num__"):
            origin = name[5:]
        elif name.startswith("cat__"):
            stripped = name[5:]
            origin = next(
                (c for c in sorted(categorical, key=len, reverse=True)
                 if stripped.startswith(c + "_")), stripped)
        totals[origin] = totals.get(origin, 0.0) + float(value)
    grand = sum(totals.values()) or 1.0
    ranked = sorted(totals.items(), key=lambda kv: -kv[1])[:15]
    return [{"feature": k, "importance": round(v / grand, 4)} for k, v in ranked]


def _selected_features(pipeline: Pipeline) -> list[str] | None:
    if "select" not in pipeline.named_steps:
        return None
    names = pipeline.named_steps["prep"].get_feature_names_out()
    mask = pipeline.named_steps["select"].get_support()
    return [str(n) for n, keep in zip(names, mask) if keep]


# ------------------------------------------------------------------------ training
def train_model(df: pd.DataFrame, request) -> TrainingResult:
    """Train and evaluate one model. ``request`` is a TrainRequest-like object."""
    settings = get_settings()
    started = time.perf_counter()
    task, algorithm = request.task, request.algorithm
    random_state = (
        request.random_state if request.random_state is not None
        else settings.ml_random_state
    )
    hp = _validate_hyperparameters(algorithm, dict(request.hyperparameters))
    _build_model(task, algorithm, dict(hp), random_state)  # validates algorithm name
    warnings: list[str] = []

    if task != "clustering":
        if not request.target:
            raise ValueError("A target column is required for this task")
        if request.target not in df.columns:
            raise ValueError(f"Target column not found: {request.target}")
    elif request.feature_selection.method != "none":
        raise ValueError("Supervised feature selection is not available for clustering")

    target = request.target if task != "clustering" else None
    work = df
    if len(work) > settings.ml_max_train_rows:
        work = work.sample(settings.ml_max_train_rows, random_state=random_state)
        warnings.append(
            f"Dataset sampled to {settings.ml_max_train_rows} rows for training."
        )

    y = None
    classes = None
    if task == "regression":
        y_all = pd.to_numeric(work[target], errors="coerce")
        if y_all.notna().sum() < 0.8 * work[target].notna().sum() or y_all.notna().sum() == 0:
            raise ValueError(f"Target '{target}' is not numeric; use classification")
        keep = y_all.notna()
        work, y = work.loc[keep], y_all.loc[keep]
        if y.nunique() < 5:
            raise ValueError(
                f"Target '{target}' has only {y.nunique()} distinct values; "
                "try classification")
        if y.std() == 0:
            raise ValueError("Target has no variance")
    elif task == "classification":
        keep = work[target].notna()
        work = work.loc[keep]
        y = work[target].astype(str)
        counts = y.value_counts()
        if len(counts) < 2:
            raise ValueError("Target needs at least two classes")
        if len(counts) > MAX_CLASSES:
            raise ValueError(
                f"Target has {len(counts)} classes (limit {MAX_CLASSES}); "
                "it looks continuous or like an identifier")
        if pd.api.types.is_float_dtype(work[target]) and work[target].nunique() > 20:
            raise ValueError(
                f"Target '{target}' looks continuous ({work[target].nunique()} "
                "distinct values); use regression")
        if counts.min() < 2:
            rare = counts[counts < 2].index.tolist()[:3]
            raise ValueError(
                f"Every class needs at least 2 rows; too rare: {', '.join(rare)}")
        if counts.min() / counts.sum() < 0.1:
            warnings.append(
                "Classes are imbalanced (smallest class below 10%); "
                "accuracy alone can be misleading, see balanced accuracy and F1.")
        classes = sorted(counts.index.tolist())

    if len(work) < settings.ml_min_rows:
        raise ValueError(
            f"At least {settings.ml_min_rows} usable rows are required "
            f"(found {len(work)})")

    numeric, categorical, features, excluded = select_features(
        work, target, request.features, task)
    for item in excluded:
        warnings.append(f"Excluded '{item['column']}': {item['reason']}.")

    X = coerce_features(work, numeric, categorical)
    prep = _preprocessor(numeric, categorical)

    if task == "clustering":
        return _train_clustering(
            work, X, prep, request, hp, numeric, categorical, features,
            warnings, excluded, random_state, started)

    steps = [("prep", prep)]
    fs = request.feature_selection
    if fs.method == "k_best":
        if not fs.k:
            raise ValueError("feature_selection.k is required for k_best")
        score_fn = f_regression if task == "regression" else f_classif
        steps.append(("select", SelectKBest(score_fn, k="all")))
    steps.append(("model", _build_model(task, algorithm, dict(hp), random_state)))
    pipeline = Pipeline(steps)

    stratify = y if task == "classification" else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=request.test_size, random_state=random_state,
            stratify=stratify)
    except ValueError as exc:
        raise ValueError(
            f"Cannot split the data: {exc}. Use more rows or a larger test_size."
        ) from None

    if fs.method == "k_best":
        # Resolve k from the TRAINING partition only (no peeking at test data).
        n_encoded = clone(prep).fit(X_train).get_feature_names_out().shape[0]
        k = min(fs.k, n_encoded)
        pipeline.set_params(select__k=k)
        if fs.k > n_encoded:
            warnings.append(f"k reduced from {fs.k} to {k} (available features).")

    pipeline.fit(X_train, y_train)
    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)

    folds = request.cv_folds
    cv = {}
    if folds >= 2 and len(X_train) >= folds * 2:
        scoring = "r2" if task == "regression" else "f1_weighted"
        try:
            scores = cross_val_score(pipeline, X_train, y_train, cv=folds,
                                     scoring=scoring, n_jobs=1)
            cv = {f"cv_{scoring}_mean": float(np.mean(scores)),
                  f"cv_{scoring}_std": float(np.std(scores)), "cv_folds": folds}
        except ValueError:
            warnings.append("Cross-validation skipped (not enough rows per class/fold).")

    if task == "regression":
        metrics, details = _regression_eval(
            pipeline, X_train, y_train, X_test, y_test, train_pred, test_pred)
    else:
        metrics, details = _classification_eval(
            pipeline, X_train, y_train, X_test, y_test, train_pred, test_pred, classes)
    metrics.update(cv)

    gap_key = "r2" if task == "regression" else "accuracy"
    train_score = details["train_score"]
    if train_score - metrics[gap_key] > 0.2:
        warnings.append(
            f"Possible overfitting: training {gap_key} {train_score:.3f} vs "
            f"test {metrics[gap_key]:.3f}.")

    importance = _importance(pipeline, numeric, categorical)
    details.update(
        {
            "split": {"train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
                      "test_size": request.test_size, "stratified": task == "classification"},
            "excluded_features": excluded,
            "feature_importance": importance,
            "selected_encoded_features": _selected_features(pipeline),
            "feature_schema": _feature_schema(work, numeric, categorical),
            "rows_used": int(len(work)),
        }
    )
    return TrainingResult(
        pipeline=pipeline, task=task, algorithm=algorithm, target=target,
        features=features, numeric_features=numeric, categorical_features=categorical,
        classes=classes, metrics=json_safe(metrics), details=json_safe(details),
        warnings=warnings, duration_ms=int((time.perf_counter() - started) * 1000),
        effective_config={"random_state": random_state, "hyperparameters": hp},
    )


def _regression_eval(pipeline, X_train, y_train, X_test, y_test, train_pred, test_pred):
    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train).predict(X_test)
    mse = mean_squared_error(y_test, test_pred)
    metrics = {
        "r2": r2_score(y_test, test_pred),
        "mae": mean_absolute_error(y_test, test_pred),
        "rmse": float(np.sqrt(mse)),
        "baseline_r2": r2_score(y_test, baseline),
        "baseline_mae": mean_absolute_error(y_test, baseline),
    }
    residuals = np.asarray(y_test) - test_pred
    sample = min(len(y_test), 200)
    idx = np.linspace(0, len(y_test) - 1, sample).astype(int)
    details = {
        "train_score": float(r2_score(y_train, train_pred)),
        "residuals": {"mean": float(residuals.mean()), "std": float(residuals.std())},
        "predictions_sample": [
            {"actual": float(np.asarray(y_test)[i]), "predicted": float(test_pred[i])}
            for i in idx
        ],
    }
    return metrics, details


def _classification_eval(pipeline, X_train, y_train, X_test, y_test, train_pred, test_pred, classes):
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train).predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, test_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, test_pred),
        "precision_weighted": precision_score(y_test, test_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, test_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, test_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_test, test_pred, average="macro", zero_division=0),
        "baseline_accuracy": accuracy_score(y_test, baseline),
    }
    try:
        proba = pipeline.predict_proba(X_test)
        if len(classes) == 2:
            metrics["roc_auc"] = roc_auc_score(y_test, proba[:, 1])
        else:
            metrics["roc_auc"] = roc_auc_score(
                y_test, proba, multi_class="ovr", average="weighted",
                labels=pipeline.classes_)
    except (ValueError, AttributeError):
        pass
    report = classification_report(y_test, test_pred, labels=classes,
                                   output_dict=True, zero_division=0)
    details = {
        "train_score": float(accuracy_score(y_train, train_pred)),
        "classes": classes,
        "confusion_matrix": confusion_matrix(y_test, test_pred, labels=classes).tolist(),
        "per_class": {
            c: {k: report[c][k] for k in ("precision", "recall", "f1-score", "support")}
            for c in classes if c in report
        },
    }
    return metrics, details


def _train_clustering(work, X, prep, request, hp, numeric, categorical, features,
                      warnings, excluded, random_state, started) -> TrainingResult:
    n_rows = len(X)
    if n_rows < 10:
        raise ValueError("Clustering needs at least 10 rows")
    Z = prep.fit_transform(X)
    max_k = min(8, n_rows - 1)
    sample_idx = (np.random.default_rng(random_state).choice(n_rows, 2000, replace=False)
                  if n_rows > 2000 else np.arange(n_rows))

    k_scores = {}
    if request.n_clusters:
        k = request.n_clusters
        if k >= n_rows:
            raise ValueError("n_clusters must be smaller than the number of rows")
    else:
        for candidate in range(2, max_k + 1):
            labels = KMeans(n_clusters=candidate, n_init=5,
                            random_state=random_state).fit_predict(Z[sample_idx])
            if len(set(labels)) > 1:
                k_scores[candidate] = float(silhouette_score(Z[sample_idx], labels))
        if not k_scores:
            raise ValueError("Could not find a valid number of clusters")
        k = max(k_scores, key=k_scores.get)

    steps = [("prep", prep), ("model", KMeans(n_clusters=k, n_init=10, random_state=random_state))]
    pipeline = Pipeline(steps).fit(X)
    labels = pipeline.named_steps["model"].labels_
    if len(set(labels)) < 2:
        raise ValueError("Clustering produced a single cluster")

    metrics = {
        "silhouette": silhouette_score(Z[sample_idx], labels[sample_idx]),
        "davies_bouldin": davies_bouldin_score(Z, labels),
        "calinski_harabasz": calinski_harabasz_score(Z, labels),
        "inertia": float(pipeline.named_steps["model"].inertia_),
        "n_clusters": int(k),
    }
    sizes = pd.Series(labels).value_counts().sort_index()
    profiles = []
    for cluster, size in sizes.items():
        members = work.iloc[np.where(labels == cluster)[0]]
        profiles.append({
            "cluster": int(cluster), "size": int(size),
            "percentage": round(size / n_rows * 100, 2),
            "numeric_means": {c: float(pd.to_numeric(members[c], errors="coerce").mean())
                              for c in numeric[:8]},
            "top_categories": {c: clip_text(members[c].mode().iloc[0], 40)
                               for c in categorical[:4] if not members[c].mode().empty},
        })
    pca_points = []
    if Z.shape[1] >= 2:
        pca = PCA(n_components=2, random_state=random_state).fit(Z)
        proj_idx = np.linspace(0, n_rows - 1, min(n_rows, 500)).astype(int)
        coords = pca.transform(Z[proj_idx])
        pca_points = [{"x": float(a), "y": float(b), "cluster": int(labels[i])}
                      for (a, b), i in zip(coords, proj_idx)]
        explained = [float(v) for v in pca.explained_variance_ratio_]
    else:
        explained = []
    details = {
        "k_selection": ({str(k_): v for k_, v in k_scores.items()}
                        if k_scores else "user-specified"),
        "cluster_profiles": profiles,
        "pca_projection": pca_points,
        "pca_explained_variance": explained,
        "excluded_features": excluded,
        "feature_importance": [],
        "feature_schema": _feature_schema(work, numeric, categorical),
        "rows_used": int(n_rows),
        "note": "Clustering is unsupervised: there is no train/test split and "
                "metrics are internal measures of cluster separation.",
    }
    return TrainingResult(
        pipeline=pipeline, task="clustering", algorithm="kmeans", target=None,
        features=features, numeric_features=numeric, categorical_features=categorical,
        classes=None, metrics=json_safe(metrics), details=json_safe(details),
        warnings=warnings, duration_ms=int((time.perf_counter() - started) * 1000),
        effective_config={"random_state": random_state, "hyperparameters": hp},
    )


def environment_info() -> dict:
    return {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
