
def test_generate_summary_handles_all_missing_numeric_values(tmp_path):
    csv_path = tmp_path / "all_missing.csv"
    pd.DataFrame({
        "score": [None, None],
    }).to_csv(csv_path, index=False)

    result = generate_summary(str(csv_path))
    stats = result["numeric_statistics"]["score"]

    assert stats["mean"] is None
    assert stats["median"] is None
    assert stats["min"] is None
    assert stats["max"] is None
    assert stats["standard_deviation"] is None


def test_generate_summary_returns_json_serializable_data(tmp_path):
    csv_path = tmp_path / "json.csv"
    pd.DataFrame({
        "score": [10, None, 30],
        "name": ["Ali", "Sara", "Ahmed"],
    }).to_csv(csv_path, index=False)

    result = generate_summary(str(csv_path))

    # Strict JSON encoding rejects NaN and Infinity.
    json.dumps(result, allow_nan=False)


def test_generate_summary_raises_for_missing_file(tmp_path):
    csv_path = tmp_path / "does_not_exist.csv"

    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        generate_summary(str(csv_path))

def test_get_dataset_summary_success():
    mock_dataset = MagicMock()
    mock_dataset.id = 1
    mock_dataset.filename = "sales.csv"
    mock_dataset.file_path = "data/raw/sales.csv"

    expected_summary = {
        "row_count": 3,
        "column_count": 2,
        "duplicate_rows": 0,
        "columns": ["sales", "region"],
        "column_details": {},
        "numeric_statistics": {},
    }

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = (
        mock_dataset
    )

    with (
        patch.object(datasets_routes, "SessionLocal", return_value=mock_db),
        patch.object(
            datasets_routes,
            "generate_summary",
            return_value=expected_summary,
        ) as mock_generate,
    ):
        response = TestClient(app).get("/datasets/1/summary")

    assert response.status_code == 200
    assert response.json() == {
        "dataset_id": 1,
        "filename": "sales.csv",
        "summary": expected_summary,
    }
    mock_generate.assert_called_once_with("data/raw/sales.csv")
    mock_db.close.assert_called_once()


def test_get_dataset_summary_dataset_not_found():
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = None

    with patch.object(
        datasets_routes, "SessionLocal", return_value=mock_db
    ):
        response = TestClient(app).get("/datasets/999/summary")

    assert response.status_code == 404
    assert response.json()["detail"] == "Dataset not found"
    mock_db.close.assert_called_once()


def test_get_dataset_summary_file_not_found():
    mock_dataset = MagicMock()
    mock_dataset.id = 1
    mock_dataset.filename = "missing.csv"
    mock_dataset.file_path = "data/raw/missing.csv"

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = (
        mock_dataset
    )

    with (
        patch.object(datasets_routes, "SessionLocal", return_value=mock_db),
        patch.object(
            datasets_routes,
            "generate_summary",
            side_effect=FileNotFoundError("Dataset file not found"),
        ),
    ):
        response = TestClient(app).get("/datasets/1/summary")

    assert response.status_code == 404
    assert response.json()["detail"] == "Dataset file not found"
    mock_db.close.assert_called_once()


def test_get_dataset_summary_invalid_csv():
    mock_dataset = MagicMock()
    mock_dataset.id = 1
    mock_dataset.filename = "invalid.csv"
    mock_dataset.file_path = "data/raw/invalid.csv"

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = (
        mock_dataset
    )

    with (
        patch.object(datasets_routes, "SessionLocal", return_value=mock_db),
        patch.object(
            datasets_routes,
            "generate_summary",
            side_effect=pd.errors.ParserError("Invalid CSV"),
        ),
    ):
        response = TestClient(app).get("/datasets/1/summary")

    assert response.status_code == 400
    assert response.json()["detail"] == "Unable to analyze dataset file"
    mock_db.close.assert_called_once()


def test_get_dataset_summary_closes_db_when_analysis_fails():
    mock_dataset = MagicMock()
    mock_dataset.id = 1
    mock_dataset.filename = "invalid.csv"
    mock_dataset.file_path = "data/raw/invalid.csv"

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = (
        mock_dataset
    )
import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.app.schemas.dataset import DatasetValidationRequest
from src.app.services.validation import validate_dataset
from src.app.models.dataset_version import DatasetVersion

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.app.main import app
from src.app.models.dataset import Dataset
from src.app.schemas.dataset import DatasetFilter
from src.app.services.profiling import generate_profile
from src.app.services.query import query_dataset
from src.app.services.summary import generate_summary
from src.app.services.visualization import generate_chart_data
import src.app.routes.datasets as datasets_routes

from src.app.schemas.dataset import (
    DatasetFilter,
    DatasetValidationRequest,
    RequiredColumnRule,
    NumericRangeRule,
    AllowedValuesRule,
)

def test_application_starts():
    message = "AI Data Platform started!"
    assert message == "AI Data Platform started!"

def test_root_endpoint():
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "AI Data Platform is running",
        "version": "0.1.0",
    }

def test_health_endpoint():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_generate_profile(tmp_path):
    csv_path = tmp_path / "test.csv"
    profile_path = tmp_path / "profile.html"

    csv_path.write_text(
        "name,age,salary\n"
        "Ali,25,50000\n"
        "Sara,30,65000\n"
        "Ahmed,28,55000\n",
        encoding="utf-8",
    )

    generate_profile(str(csv_path), str(profile_path))

    assert Path(profile_path).exists()

def test_list_datasets():
    mock_dataset = Dataset(
        id=101,
        filename="sample.csv",
        profile_path="data/profiles/sample_profile.html",
        row_count=3,
        column_count=3,
        status="uploaded",
    )

    mock_db = MagicMock()
    mock_db.query.return_value.order_by.return_value.all.return_value = [
        mock_dataset
    ]

    with patch("src.app.routes.datasets.SessionLocal", return_value=mock_db):
        client = TestClient(app)
        response = client.get("/datasets/")

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": 101,
            "filename": "sample.csv",
            "profile_path": "data/profiles/sample_profile.html",
            "row_count": 3,
            "column_count": 3,
            "status": "uploaded",
        }
    ]

    mock_db.close.assert_called_once()

def test_get_dataset():
    mock_dataset = Dataset(
        id=101,
        filename="sample.csv",
        profile_path="data/profiles/sample_profile.html",
        row_count=3,
        column_count=3,
        status="uploaded",
    )

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = (
        mock_dataset
    )

    with patch(
        "src.app.routes.datasets.SessionLocal",
        return_value=mock_db,
    ):
        client = TestClient(app)
        response = client.get("/datasets/101")

    assert response.status_code == 200
    assert response.json() == {
        "id": 101,
        "filename": "sample.csv",
        "profile_path": "data/profiles/sample_profile.html",
        "row_count": 3,
        "column_count": 3,
        "status": "uploaded",
    }

    mock_db.close.assert_called_once()

def test_get_dataset_not_found():
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = None

    with patch(
        "src.app.routes.datasets.SessionLocal",
        return_value=mock_db,
    ):
        client = TestClient(app)
        response = client.get("/datasets/999999")

    assert response.status_code == 404
    assert response.json() == {"detail": "Dataset not found"}

    mock_db.close.assert_called_once()

def test_delete_dataset(tmp_path):
    dataset_file = tmp_path / "sample.csv"
    profile_file = tmp_path / "sample_profile.html"
    version_file = tmp_path / "sample_v2.csv"

    dataset_file.write_text("name,age\nAlice,25")
    profile_file.write_text("<html>Profile</html>")
    version_file.write_text("name,age\nBob,30")

    mock_dataset = Dataset(
        id=101,
        filename="sample.csv",
        file_path=str(dataset_file),
        profile_path=str(profile_file),
        row_count=1,
        column_count=2,
        status="uploaded",
    )

    mock_version = DatasetVersion(
        id=1,
        dataset_id=101,
        version_number=1,
        file_path=str(dataset_file),
    )

    mock_version_2 = DatasetVersion(
        id=2,
        dataset_id=101,
        version_number=2,
        file_path=str(version_file),
    )

    mock_db = MagicMock()

    mock_db.query.return_value.filter.return_value.first.return_value = (
        mock_dataset
    )

    mock_db.query.return_value.filter.return_value.all.return_value = [
        mock_version,
        mock_version_2,
    ]

    with patch(
        "src.app.routes.datasets.SessionLocal",
        return_value=mock_db,
    ):
        client = TestClient(app)
        response = client.delete("/datasets/101")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Dataset deleted successfully",
        "id": 101,
    }

    mock_db.delete.assert_called_once_with(mock_dataset)
    mock_db.commit.assert_called_once()
    mock_db.close.assert_called_once()

    assert not dataset_file.exists()
    assert not profile_file.exists()
    assert not version_file.exists()

def test_delete_dataset_not_found():
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = None

    with patch(
        "src.app.routes.datasets.SessionLocal",
        return_value=mock_db,
    ):
        client = TestClient(app)
        response = client.delete("/datasets/999999")

    assert response.status_code == 404
    assert response.json() == {"detail": "Dataset not found"}

    mock_db.delete.assert_not_called()
    mock_db.commit.assert_not_called()
    mock_db.close.assert_called_once()


def test_get_dataset_closes_db_when_not_found():
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = None

    with patch(
        "src.app.routes.datasets.SessionLocal",
        return_value=mock_db,
    ):
        client = TestClient(app)
        response = client.get("/datasets/999999")

    assert response.status_code == 404
    mock_db.close.assert_called_once()


def test_upload_dataset_success(tmp_path):
    csv_content = b"name,age,salary\nAli,25,50000\nSara,30,65000\n"

    mock_db = MagicMock()

    def flush_dataset():
        dataset = mock_db.add.call_args.args[0]
        dataset.id = 201

    mock_db.flush.side_effect = flush_dataset

    def refresh_dataset(dataset):
        dataset.id = 201

    mock_db.refresh.side_effect = refresh_dataset

    with (
        patch.object(datasets_routes, "UPLOAD_DIR", tmp_path),
        patch.object(datasets_routes, "PROFILE_DIR", tmp_path),
        patch.object(
            datasets_routes,
            "SessionLocal",
            return_value=mock_db,
        ),
        patch.object(
            datasets_routes,
            "generate_profile",
        ) as mock_profile,
    ):
        client = TestClient(app)
        response = client.post(
            "/datasets/upload",
            files={
                "file": (
                    "sample.csv",
                    BytesIO(csv_content),
                    "text/csv",
                )
            },
        )

    assert response.status_code == 200

    result = response.json()
    assert result["id"] == 201
    assert result["filename"] == "sample.csv"
    assert result["row_count"] == 2
    assert result["column_count"] == 3
    assert result["profile_path"].endswith("_profile.html")

    assert mock_db.add.call_count == 2

    added_dataset = mock_db.add.call_args_list[0].args[0]
    added_version = mock_db.add.call_args_list[1].args[0]

    assert added_dataset.id == 201
    assert added_version.dataset_id == 201
    assert added_version.version_number == 1
    assert added_version.row_count == 2
    assert added_version.column_count == 3

    mock_db.commit.assert_called_once()
    mock_db.close.assert_called_once()
    mock_profile.assert_called_once()

def test_upload_dataset_version_success(tmp_path):
    csv_content = b"name,age,salary\nAli,25,50000\nSara,30,65000\n"

    mock_dataset = SimpleNamespace(
        id=201,
        filename="old.csv",
        file_path="old.csv",
        profile_path="old_profile.html",
        row_count=1,
        column_count=2,
    )

    latest_version = SimpleNamespace(version_number=1)

    mock_db = MagicMock()

    dataset_query = MagicMock()
    dataset_query.filter.return_value.first.return_value = mock_dataset

    version_query = MagicMock()
    version_query.filter.return_value.order_by.return_value.first.return_value = (
        latest_version
    )

    mock_db.query.side_effect = [dataset_query, version_query]

    def refresh_version(version):
        version.id = 301

    mock_db.refresh.side_effect = refresh_version

    with (
        patch.object(datasets_routes, "UPLOAD_DIR", tmp_path),
        patch.object(datasets_routes, "PROFILE_DIR", tmp_path),
        patch.object(
            datasets_routes,
            "SessionLocal",
            return_value=mock_db,
        ),
        patch.object(
            datasets_routes,
            "generate_profile",
        ) as mock_profile,
    ):
        client = TestClient(app)
        response = client.post(
            "/datasets/201/versions/upload",
            files={
                "file": (
                    "updated.csv",
                    BytesIO(csv_content),
                    "text/csv",
                )
            },
        )

    assert response.status_code == 200

    result = response.json()
    assert result["dataset_id"] == 201
    assert result["version_id"] == 301
    assert result["version_number"] == 2
    assert result["filename"] == "updated.csv"
    assert result["row_count"] == 2
    assert result["column_count"] == 3

    added_version = mock_db.add.call_args.args[0]

    assert added_version.dataset_id == 201
    assert added_version.version_number == 2
    assert added_version.row_count == 2
    assert added_version.column_count == 3

    assert mock_dataset.filename == "updated.csv"
    assert mock_dataset.row_count == 2
    assert mock_dataset.column_count == 3

    mock_db.commit.assert_called_once()
    mock_db.close.assert_called_once()
    mock_profile.assert_called_once()


def test_upload_dataset_version_not_found(tmp_path):
    csv_content = b"name,age\nAli,25\n"

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = None

    with (
        patch.object(datasets_routes, "UPLOAD_DIR", tmp_path),
        patch.object(datasets_routes, "PROFILE_DIR", tmp_path),
        patch.object(
            datasets_routes,
            "SessionLocal",
            return_value=mock_db,
        ),
        patch.object(datasets_routes, "generate_profile"),
    ):
        client = TestClient(app)
        response = client.post(
            "/datasets/999/versions/upload",
            files={
                "file": (
                    "sample.csv",
                    BytesIO(csv_content),
                    "text/csv",
                )
            },
        )

    assert response.status_code == 404
    assert response.json() == {"detail": "Dataset not found"}
    mock_db.close.assert_called_once()


def test_upload_dataset_version_invalid_csv(tmp_path):
    csv_content = b'name,age\n"Ali,25\n'

    with (
        patch.object(datasets_routes, "UPLOAD_DIR", tmp_path),
        patch.object(datasets_routes, "PROFILE_DIR", tmp_path),
        patch.object(
            datasets_routes,
            "SessionLocal",
        ) as mock_session,
    ):
        client = TestClient(app)
        response = client.post(
            "/datasets/201/versions/upload",
            files={
                "file": (
                    "invalid.csv",
                    BytesIO(csv_content),
                    "text/csv",
                )
            },
        )

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid CSV file"}
    mock_session.assert_not_called()


def test_upload_dataset_version_empty_csv(tmp_path):
    with (
        patch.object(datasets_routes, "UPLOAD_DIR", tmp_path),
        patch.object(datasets_routes, "PROFILE_DIR", tmp_path),
        patch.object(
            datasets_routes,
            "SessionLocal",
        ) as mock_session,
    ):
        client = TestClient(app)
        response = client.post(
            "/datasets/201/versions/upload",
            files={
                "file": (
                    "empty.csv",
                    BytesIO(b""),
                    "text/csv",
                )
            },
        )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Uploaded CSV file is empty"
    }
    mock_session.assert_not_called()

def test_upload_dataset_invalid_csv(tmp_path):
    mock_db = MagicMock()

    with (
        patch.object(datasets_routes, "UPLOAD_DIR", tmp_path),
        patch.object(datasets_routes, "PROFILE_DIR", tmp_path),
        patch.object(datasets_routes, "SessionLocal", return_value=mock_db),
    ):
        client = TestClient(app)
        response = client.post(
            "/datasets/upload",
            files={
                "file": (
                    "invalid.csv",
                    BytesIO(b'\xff\xfe\x00\x00'),
                    "text/csv",
                )
            },
        )

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid CSV file"}
    mock_db.add.assert_not_called()
    mock_db.close.assert_not_called()


def test_upload_dataset_empty_csv(tmp_path):
    mock_db = MagicMock()

    with (
        patch.object(datasets_routes, "UPLOAD_DIR", tmp_path),
        patch.object(datasets_routes, "PROFILE_DIR", tmp_path),
        patch.object(datasets_routes, "SessionLocal", return_value=mock_db),
    ):
        client = TestClient(app)
        response = client.post(
            "/datasets/upload",
            files={
                "file": ("empty.csv", BytesIO(b"  \n"), "text/csv")
            },
        )

    assert response.status_code == 400
    assert response.json() == {"detail": "Uploaded CSV file is empty"}
    mock_db.add.assert_not_called()


def test_upload_dataset_rejects_non_csv(tmp_path):
    mock_db = MagicMock()

    with (
        patch.object(datasets_routes, "UPLOAD_DIR", tmp_path),
        patch.object(datasets_routes, "PROFILE_DIR", tmp_path),
        patch.object(datasets_routes, "SessionLocal", return_value=mock_db),
    ):
        client = TestClient(app)
        response = client.post(
            "/datasets/upload",
            files={
                "file": (
                    "notes.txt",
                    BytesIO(b"hello"),
                    "text/plain",
                )
            },
        )

    assert response.status_code == 400
    assert response.json() == {"detail": "Only CSV files are supported"}
    mock_db.add.assert_not_called()


def test_upload_dataset_missing_file():
    client = TestClient(app)
    response = client.post("/datasets/upload")

    assert response.status_code == 422


def test_upload_dataset_profiling_failure(tmp_path):
    mock_db = MagicMock()

    with (
        patch.object(datasets_routes, "UPLOAD_DIR", tmp_path),
        patch.object(datasets_routes, "PROFILE_DIR", tmp_path),
        patch.object(datasets_routes, "SessionLocal", return_value=mock_db),
        patch.object(
            datasets_routes,
            "generate_profile",
            side_effect=RuntimeError("Profiling failed"),
        ),
    ):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/datasets/upload",
            files={
                "file": (
                    "sample.csv",
                    BytesIO(b"name,age\nAli,25\n"),
                    "text/csv",
                )
            },
        )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Failed to process uploaded dataset"
    }
    mock_db.add.assert_not_called()
    mock_db.close.assert_not_called()


def test_list_datasets_empty():
    mock_db = MagicMock()
    mock_db.query.return_value.order_by.return_value.all.return_value = []

    with patch(
        "src.app.routes.datasets.SessionLocal",
        return_value=mock_db,
    ):
        client = TestClient(app)
        response = client.get("/datasets/")

    assert response.status_code == 200
    assert response.json() == []
    mock_db.close.assert_called_once()

def test_get_dataset_profile_success(tmp_path):
    profile_file = tmp_path / "sample_profile.html"
    profile_file.write_text(
        "<html><body>Test profile</body></html>",
        encoding="utf-8",
    )

    mock_dataset = Dataset(
        id=101,
        filename="sample.csv",
        profile_path=str(profile_file),
        row_count=3,
        column_count=3,
        status="uploaded",
    )

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = (
        mock_dataset
    )

    with patch(
        "src.app.routes.datasets.SessionLocal",
        return_value=mock_db,
    ):
        client = TestClient(app)
        response = client.get("/datasets/101/profile")

    assert response.status_code == 200
    assert response.text == "<html><body>Test profile</body></html>"
    assert "text/html" in response.headers["content-type"]
    mock_db.close.assert_called_once()


def test_get_dataset_profile_dataset_not_found():
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = None

    with patch(
        "src.app.routes.datasets.SessionLocal",
        return_value=mock_db,
    ):
        client = TestClient(app)
        response = client.get("/datasets/999999/profile")

    assert response.status_code == 404
    assert response.json() == {"detail": "Dataset not found"}
    mock_db.close.assert_called_once()


def test_get_dataset_profile_file_missing(tmp_path):
    missing_profile = tmp_path / "missing_profile.html"

    mock_dataset = Dataset(
        id=101,
        filename="sample.csv",
        profile_path=str(missing_profile),
        row_count=3,
        column_count=3,
        status="uploaded",
    )

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = (
        mock_dataset
    )

    with patch(
        "src.app.routes.datasets.SessionLocal",
        return_value=mock_db,
    ):
        client = TestClient(app)
        response = client.get("/datasets/101/profile")

    assert response.status_code == 404
    assert response.json() == {"detail": "Profile report not found"}
    mock_db.close.assert_called_once()

def test_generate_summary_returns_dataset_overview(tmp_path):
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame({
        "age": [20, 30, 40],
        "name": ["Ali", "Sara", "Ahmed"],
    }).to_csv(csv_path, index=False)

    result = generate_summary(str(csv_path))

    assert result["row_count"] == 3
    assert result["column_count"] == 2
    assert result["duplicate_rows"] == 0
    assert result["columns"] == ["age", "name"]


def test_generate_summary_returns_numeric_statistics(tmp_path):
    csv_path = tmp_path / "numbers.csv"
    pd.DataFrame({
        "score": [10, 20, 30, 40],
    }).to_csv(csv_path, index=False)

    result = generate_summary(str(csv_path))
    stats = result["numeric_statistics"]["score"]

    assert stats["mean"] == 25.0
    assert stats["median"] == 25.0
    assert stats["min"] == 10.0
    assert stats["max"] == 40.0
    assert stats["standard_deviation"] == pytest.approx(12.909944)


def test_generate_summary_counts_missing_values(tmp_path):
    csv_path = tmp_path / "missing.csv"
    pd.DataFrame({
        "age": [20, None, 40, None],
        "name": ["Ali", "Sara", None, "Ahmed"],
    }).to_csv(csv_path, index=False)

    result = generate_summary(str(csv_path))

    assert result["column_details"]["age"]["missing_values"] == 2
    assert result["column_details"]["age"]["missing_percentage"] == 50.0
    assert result["column_details"]["name"]["missing_values"] == 1
    assert result["column_details"]["name"]["missing_percentage"] == 25.0


def test_generate_summary_counts_unique_values(tmp_path):
    csv_path = tmp_path / "unique.csv"
    pd.DataFrame({
        "city": ["Lahore", "Karachi", "Lahore", None],
    }).to_csv(csv_path, index=False)

    result = generate_summary(str(csv_path))

    assert result["column_details"]["city"]["unique_values"] == 2


def test_generate_summary_counts_duplicate_rows(tmp_path):
    csv_path = tmp_path / "duplicates.csv"
    pd.DataFrame({
        "id": [1, 2, 2, 3],
        "city": ["Lahore", "Karachi", "Karachi", "Islamabad"],
    }).to_csv(csv_path, index=False)

    result = generate_summary(str(csv_path))

    assert result["duplicate_rows"] == 1

def test_generate_summary_handles_empty_dataset(tmp_path):
    csv_path = tmp_path / "empty.csv"
    pd.DataFrame(columns=["age", "name"]).to_csv(csv_path, index=False)

    result = generate_summary(str(csv_path))

    assert result["row_count"] == 0
    assert result["column_count"] == 2
    assert result["duplicate_rows"] == 0
    assert result["column_details"]["age"]["missing_percentage"] == 0.0


def test_summary_endpoint_closes_db_on_unexpected_error():
    dataset = SimpleNamespace(
        id=1,
        filename="sample.csv",
        file_path="sample.csv",
    )

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = dataset

    with (
        patch.object(
            datasets_routes,
            "SessionLocal",
            return_value=mock_db,
        ),
        patch.object(
            datasets_routes,
            "generate_summary",
            side_effect=RuntimeError("Unexpected analysis failure"),
        ),
        pytest.raises(RuntimeError, match="Unexpected analysis failure"),
    ):
        TestClient(app).get("/datasets/1/summary")

    mock_db.close.assert_called_once()


def test_generate_histogram_chart_data(tmp_path):
    csv_path = tmp_path / "scores.csv"
    pd.DataFrame({"score": [10, 20, 30, 40]}).to_csv(
        csv_path, index=False
    )

    result = generate_chart_data(
        str(csv_path), "histogram", "score", bins=2
    )

    assert result["chart_type"] == "histogram"
    assert result["x_column"] == "score"
    assert len(result["bins"]) == 2
    assert sum(item["count"] for item in result["bins"]) == 4


def test_generate_bar_chart_data(tmp_path):
    csv_path = tmp_path / "cities.csv"
    pd.DataFrame({
        "city": ["Lahore", "Karachi", "Lahore", "Islamabad", None],
    }).to_csv(csv_path, index=False)

    result = generate_chart_data(str(csv_path), "bar", "city")

    assert result["chart_type"] == "bar"
    assert result["x_column"] == "city"

    counts = {item["category"]: item["count"] for item in result["data"]}
    assert counts["Lahore"] == 2
    assert counts["Karachi"] == 1
    assert counts["(Missing)"] == 1


def test_generate_scatter_chart_data(tmp_path):
    csv_path = tmp_path / "measurements.csv"
    pd.DataFrame({
        "height": [150, 160, 170],
        "weight": [50, 60, 70],
    }).to_csv(csv_path, index=False)

    result = generate_chart_data(
        str(csv_path), "scatter", "height", y_column="weight"
    )

    assert result["chart_type"] == "scatter"
    assert result["x_column"] == "height"
    assert result["y_column"] == "weight"
    assert result["data"] == [
        {"x": 150.0, "y": 50.0},
        {"x": 160.0, "y": 60.0},
        {"x": 170.0, "y": 70.0},
    ]


def test_histogram_rejects_non_numeric_column(tmp_path):
    csv_path = tmp_path / "text.csv"
    pd.DataFrame({"name": ["Ali", "Sara"]}).to_csv(
        csv_path, index=False
    )

    with pytest.raises(ValueError, match="no valid numeric values"):
        generate_chart_data(str(csv_path), "histogram", "name")


@pytest.mark.parametrize("bins", [0, -1, 101])
def test_histogram_rejects_invalid_bin_count(tmp_path, bins):
    csv_path = tmp_path / "scores.csv"
    pd.DataFrame({"score": [10, 20, 30]}).to_csv(
        csv_path, index=False
    )

    with pytest.raises(ValueError, match="bins must be between 1 and 100"):
        generate_chart_data(
            str(csv_path), "histogram", "score", bins=bins
        )


def test_visualization_rejects_unsupported_chart_type(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"score": [10, 20]}).to_csv(
        csv_path, index=False
    )

    with pytest.raises(ValueError, match="Unsupported chart type"):
        generate_chart_data(str(csv_path), "pie", "score")


def test_visualization_rejects_missing_column(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"score": [10, 20]}).to_csv(
        csv_path, index=False
    )

    with pytest.raises(ValueError, match="Column not found: age"):
        generate_chart_data(str(csv_path), "bar", "age")


def test_scatter_requires_y_column(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"score": [10, 20]}).to_csv(
        csv_path, index=False
    )

    with pytest.raises(ValueError, match="y_column is required"):
        generate_chart_data(str(csv_path), "scatter", "score")


def test_scatter_rejects_no_valid_numeric_pairs(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({
        "x": ["a", "b"],
        "y": ["c", "d"],
    }).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="No valid numeric pairs"):
        generate_chart_data(
            str(csv_path), "scatter", "x", y_column="y"
        )


def test_visualization_rejects_missing_file(tmp_path):
    csv_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        generate_chart_data(str(csv_path), "bar", "city")


def test_bar_chart_limits_categories_to_twenty(tmp_path):
    csv_path = tmp_path / "categories.csv"
    pd.DataFrame({
        "category": [f"category_{i}" for i in range(25)]
    }).to_csv(csv_path, index=False)

    result = generate_chart_data(str(csv_path), "bar", "category")

    assert len(result["data"]) == 20



from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import src.app.routes.datasets as datasets_routes
from src.app.main import app


visualization_client = TestClient(app)


def test_get_dataset_visualization_success():
    dataset = SimpleNamespace(
        id=1,
        filename="sample.csv",
        file_path="data/raw/sample.csv",
    )
    chart_data = {
        "chart_type": "bar",
        "x_column": "city",
        "data": [{"category": "Lahore", "count": 3}],
    }

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = dataset

    with (
        patch.object(datasets_routes, "SessionLocal", return_value=mock_db),
        patch.object(
            datasets_routes,
            "generate_chart_data",
            return_value=chart_data,
        ) as mock_generate,
    ):
        response = visualization_client.get(
            "/datasets/1/visualization",
            params={"chart_type": "bar", "x_column": "city"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "dataset_id": 1,
        "filename": "sample.csv",
        "visualization": chart_data,
    }
    mock_generate.assert_called_once_with(
        file_path="data/raw/sample.csv",
        chart_type="bar",
        x_column="city",
        y_column=None,
        bins=10,
    )
    mock_db.close.assert_called_once()


def test_get_dataset_visualization_dataset_not_found():
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = None

    with patch.object(
        datasets_routes, "SessionLocal", return_value=mock_db
    ):
        response = visualization_client.get(
            "/datasets/999/visualization",
            params={"chart_type": "bar", "x_column": "city"},
        )

    assert response.status_code == 404
    assert response.json()["detail"] == "Dataset not found"
    mock_db.close.assert_called_once()


def test_get_dataset_visualization_file_not_found():
    dataset = SimpleNamespace(
        id=1,
        filename="sample.csv",
        file_path="missing.csv",
    )
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = dataset

    with (
        patch.object(datasets_routes, "SessionLocal", return_value=mock_db),
        patch.object(
            datasets_routes,
            "generate_chart_data",
            side_effect=FileNotFoundError("Dataset file not found"),
        ),
    ):
        response = visualization_client.get(
            "/datasets/1/visualization",
            params={"chart_type": "bar", "x_column": "city"},
        )

    assert response.status_code == 404
    assert response.json()["detail"] == "Dataset file not found"
    mock_db.close.assert_called_once()


def test_get_dataset_visualization_invalid_parameters():
    dataset = SimpleNamespace(
        id=1,
        filename="sample.csv",
        file_path="sample.csv",
    )
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = dataset

    with (
        patch.object(datasets_routes, "SessionLocal", return_value=mock_db),
        patch.object(
            datasets_routes,
            "generate_chart_data",
            side_effect=ValueError("Column not found: unknown"),
        ),
    ):
        response = visualization_client.get(
            "/datasets/1/visualization",
            params={"chart_type": "bar", "x_column": "unknown"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Column not found: unknown"
    mock_db.close.assert_called_once()


@pytest.mark.parametrize(
    "read_error",
    [
        pd.errors.ParserError("Malformed CSV"),
        UnicodeDecodeError(
            "utf-8", b"\xff", 0, 1, "invalid start byte"
        ),
    ],
)
def test_get_dataset_visualization_unreadable_file(read_error):
    dataset = SimpleNamespace(
        id=1,
        filename="sample.csv",
        file_path="sample.csv",
    )
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = dataset

    with (
        patch.object(datasets_routes, "SessionLocal", return_value=mock_db),
        patch.object(
            datasets_routes,
            "generate_chart_data",
            side_effect=read_error,
        ),
    ):
        response = visualization_client.get(
            "/datasets/1/visualization",
            params={"chart_type": "bar", "x_column": "city"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unable to read dataset file"
    mock_db.close.assert_called_once()


def test_export_dataset_success(tmp_path):
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("city,population\nLahore,13000000\n", encoding="utf-8")

    dataset = SimpleNamespace(
        id=1,
        filename="sample.csv",
        file_path=str(csv_file),
    )
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = dataset

    with patch.object(
        datasets_routes, "SessionLocal", return_value=mock_db
    ):
        response = visualization_client.get("/datasets/1/export")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "sample.csv" in response.headers["content-disposition"]

    assert response.text.replace("\r\n", "\n") == (
        "city,population\nLahore,13000000\n"
    )

    mock_db.close.assert_called_once()


def test_export_dataset_not_found():
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = None

    with patch.object(
        datasets_routes, "SessionLocal", return_value=mock_db
    ):
        response = visualization_client.get("/datasets/999/export")

    assert response.status_code == 404
    assert response.json()["detail"] == "Dataset not found"
    mock_db.close.assert_called_once()


def test_export_dataset_file_not_found(tmp_path):
    dataset = SimpleNamespace(
        id=1,
        filename="missing.csv",
        file_path=str(tmp_path / "missing.csv"),
    )
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = dataset

    with patch.object(
        datasets_routes, "SessionLocal", return_value=mock_db
    ):
        response = visualization_client.get("/datasets/1/export")

    assert response.status_code == 404
    assert response.json()["detail"] == "Dataset file not found"
    mock_db.close.assert_called_once()



from src.app.services.query import query_dataset
from src.app.schemas.dataset import DatasetFilter


def test_query_dataset_numeric_filter(tmp_path):
    csv_file = tmp_path / "people.csv"
    csv_file.write_text(
        "name,age\nAlice,25\nBob,40\nCharlie,35\n",
        encoding="utf-8",
    )

    result = query_dataset(
        str(csv_file),
        [DatasetFilter(column="age", operator="gt", value=30)],
    )

    assert result["total_matches"] == 2
    assert {row["name"] for row in result["rows"]} == {"Bob", "Charlie"}


def test_query_dataset_combines_filters_with_and(tmp_path):
    csv_file = tmp_path / "people.csv"
    csv_file.write_text(
        "name,age,city\nAlice,25,Lahore\nBob,40,Lahore\nCharlie,35,Karachi\n",
        encoding="utf-8",
    )

    result = query_dataset(
        str(csv_file),
        [
            DatasetFilter(column="age", operator="gte", value=30),
            DatasetFilter(column="city", operator="eq", value="Lahore"),
        ],
    )

    assert result["total_matches"] == 1
    assert result["rows"][0]["name"] == "Bob"


def test_query_dataset_pagination(tmp_path):
    csv_file = tmp_path / "people.csv"
    csv_file.write_text(
        "name\nAlice\nBob\nCharlie\n",
        encoding="utf-8",
    )

    result = query_dataset(str(csv_file), [], limit=1, offset=1)

    assert result["total_matches"] == 3
    assert result["returned_rows"] == 1
    assert result["rows"] == [{"name": "Bob"}]


@pytest.mark.parametrize(
    ("operator", "value", "expected"),
    [
        ("eq", "Alice", ["Alice"]),
        ("ne", "Alice", ["Charlie"]),
        ("contains", "ali", ["Alice"]),
        ("in", ["Alice", "Charlie"], ["Alice", "Charlie"]),
        ("is_null", None, [None]),
        ("not_null", None, ["Alice", "Charlie"]),
    ],
)
def test_query_dataset_string_and_null_operators(
    tmp_path, operator, value, expected
):
    csv_file = tmp_path / "people.csv"

    # Keep the row intact even when the name is missing.
    pd.DataFrame({
        "name": ["Alice", None, "Charlie"],
        "id": [1, 2, 3],
    }).to_csv(csv_file, index=False)

    result = query_dataset(
        str(csv_file),
        [DatasetFilter(column="name", operator=operator, value=value)],
    )

    assert [row["name"] for row in result["rows"]] == expected


def test_query_dataset_invalid_column(tmp_path):
    csv_file = tmp_path / "people.csv"
    csv_file.write_text("name\nAlice\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Column not found"):
        query_dataset(
            str(csv_file),
            [DatasetFilter(column="unknown", operator="eq", value="Alice")],
        )


@pytest.mark.parametrize(
    ("operator", "value"),
    [
        ("contains", 123),
        ("in", "Alice"),
        ("gt", "not-a-number"),
        ("eq", None),
    ],
)
def test_query_dataset_invalid_filter_values(tmp_path, operator, value):
    csv_file = tmp_path / "people.csv"
    csv_file.write_text("name,age\nAlice,25\n", encoding="utf-8")

    column = "age" if operator == "gt" else "name"

    with pytest.raises(ValueError):
        query_dataset(
            str(csv_file),
            [DatasetFilter(column=column, operator=operator, value=value)],
        )


def test_query_dataset_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        query_dataset(str(tmp_path / "missing.csv"), [])


def test_query_dataset_api_success(tmp_path):
    csv_file = tmp_path / "people.csv"
    csv_file.write_text("name,age\nAlice,25\nBob,40\n", encoding="utf-8")

    dataset = SimpleNamespace(
        id=1,
        filename="people.csv",
        file_path=str(csv_file),
    )
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = dataset

    with patch.object(
        datasets_routes, "SessionLocal", return_value=mock_db
    ):
        response = visualization_client.post(
            "/datasets/1/query",
            json={
                "filters": [
                    {"column": "age", "operator": "gt", "value": 30}
                ],
                "limit": 10,
                "offset": 0,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["dataset_id"] == 1
    assert body["total_matches"] == 1
    assert body["rows"] == [{"name": "Bob", "age": 40}]
    mock_db.close.assert_called_once()


def test_query_dataset_api_dataset_not_found():
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = None

    with patch.object(
        datasets_routes, "SessionLocal", return_value=mock_db
    ):
        response = visualization_client.post(
            "/datasets/999/query",
            json={"filters": []},
        )

    assert response.status_code == 404
    assert response.json()["detail"] == "Dataset not found"
    mock_db.close.assert_called_once()


def test_query_dataset_api_missing_file(tmp_path):
    dataset = SimpleNamespace(
        id=1,
        filename="missing.csv",
        file_path=str(tmp_path / "missing.csv"),
    )
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = dataset

    with patch.object(
        datasets_routes, "SessionLocal", return_value=mock_db
    ):
        response = visualization_client.post(
            "/datasets/1/query",
            json={"filters": []},
        )

    assert response.status_code == 404
    assert response.json()["detail"] == "Dataset file not found"
    mock_db.close.assert_called_once()


def test_query_dataset_api_invalid_request():
    response = visualization_client.post(
        "/datasets/1/query",
        json={"filters": [], "limit": 5000},
    )

    assert response.status_code == 422

from src.app.services.quality import generate_quality_report


def test_quality_report_detects_missing_values_and_duplicates(tmp_path):
    csv_file = tmp_path / "quality.csv"
    pd.DataFrame({
        "name": ["Alice", "Bob", "Bob", "Charlie"],
        "age": [25, None, None, 35],
        "unused": [None, None, None, None],
    }).to_csv(csv_file, index=False)

    result = generate_quality_report(str(csv_file))

    assert result["row_count"] == 4
    assert result["column_count"] == 3
    assert result["missing_cells"] == 6
    assert result["duplicate_rows"] == 1
    assert result["empty_columns"] == ["unused"]
    assert result["column_quality"]["age"]["missing_values"] == 2
    assert result["column_quality"]["age"]["missing_percentage"] == 50.0
    assert 0 <= result["quality_score"] <= 100
    assert result["warnings"]
    assert result["recommendations"]


def test_quality_report_perfect_dataset_scores_100(tmp_path):
    csv_file = tmp_path / "perfect.csv"
    pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
    }).to_csv(csv_file, index=False)

    result = generate_quality_report(str(csv_file))

    assert result["quality_score"] == 100.0
    assert result["missing_cells"] == 0
    assert result["duplicate_rows"] == 0
    assert result["empty_columns"] == []
    assert result["warnings"] == []


def test_quality_report_handles_empty_dataset(tmp_path):
    csv_file = tmp_path / "empty.csv"
    pd.DataFrame(columns=["name", "age"]).to_csv(
        csv_file, index=False
    )

    result = generate_quality_report(str(csv_file))

    assert result["row_count"] == 0
    assert result["column_count"] == 2
    assert result["quality_score"] == 0.0
    assert result["empty_columns"] == ["name", "age"]
    assert result["warnings"]


def test_quality_report_handles_missing_file(tmp_path):
    with pytest.raises(
        FileNotFoundError,
        match="Dataset file not found",
    ):
        generate_quality_report(str(tmp_path / "missing.csv"))


def test_quality_report_rejects_csv_without_columns(tmp_path):
    csv_file = tmp_path / "no_columns.csv"
    csv_file.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="no columns"):
        generate_quality_report(str(csv_file))


def test_quality_report_is_json_serializable(tmp_path):
    csv_file = tmp_path / "json.csv"
    pd.DataFrame({
        "score": [10, None, 30],
        "name": ["Alice", "Bob", "Charlie"],
    }).to_csv(csv_file, index=False)

    result = generate_quality_report(str(csv_file))

    json.dumps(result, allow_nan=False)


def test_dataset_quality_endpoint_success(tmp_path):
    csv_file = tmp_path / "sample.csv"
    pd.DataFrame({
        "name": ["Alice", "Bob"],
        "age": [25, 30],
    }).to_csv(csv_file, index=False)

    dataset = SimpleNamespace(
        id=1,
        filename="sample.csv",
        file_path=str(csv_file),
    )

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = (
        dataset
    )

    with patch.object(
        datasets_routes,
        "SessionLocal",
        return_value=mock_db,
    ):
        response = TestClient(app).get("/datasets/1/quality")

    assert response.status_code == 200
    body = response.json()
    assert body["dataset_id"] == 1
    assert body["filename"] == "sample.csv"
    assert body["quality"]["quality_score"] == 100.0
    mock_db.close.assert_called_once()


def test_dataset_quality_endpoint_dataset_not_found():
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = None

    with patch.object(
        datasets_routes,
        "SessionLocal",
        return_value=mock_db,
    ):
        response = TestClient(app).get("/datasets/999/quality")

    assert response.status_code == 404
    assert response.json()["detail"] == "Dataset not found"
    mock_db.close.assert_called_once()


def test_dataset_quality_endpoint_file_not_found(tmp_path):
    dataset = SimpleNamespace(
        id=1,
        filename="missing.csv",
        file_path=str(tmp_path / "missing.csv"),
    )

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = (
        dataset
    )

    with patch.object(
        datasets_routes,
        "SessionLocal",
        return_value=mock_db,
    ):
        response = TestClient(app).get("/datasets/1/quality")

    assert response.status_code == 404
    assert response.json()["detail"] == "Dataset file not found"
    mock_db.close.assert_called_once()


def test_dataset_quality_endpoint_closes_db_on_unexpected_error():
    dataset = SimpleNamespace(
        id=1,
        filename="sample.csv",
        file_path="sample.csv",
    )

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = (
        dataset
    )

    with (
        patch.object(
            datasets_routes,
            "SessionLocal",
            return_value=mock_db,
        ),
        patch.object(
            datasets_routes,
            "generate_quality_report",
            side_effect=RuntimeError("Unexpected failure"),
        ),
        pytest.raises(RuntimeError, match="Unexpected failure"),
    ):
        TestClient(app).get("/datasets/1/quality")

    mock_db.close.assert_called_once()

# Dataset validation service tests

def test_validate_dataset_all_rules_pass(tmp_path):
    csv_file = tmp_path / "valid.csv"
    pd.DataFrame({
        "name": ["Alice", "Bob"],
        "age": [25, 35],
        "status": ["active", "inactive"],
    }).to_csv(csv_file, index=False)

    rules = [
        RequiredColumnRule(type="required_column", column="name"),
        NumericRangeRule(
            type="numeric_range",
            column="age",
            minimum=18,
            maximum=65,
        ),
        AllowedValuesRule(
            type="allowed_values",
            column="status",
            values=["active", "inactive"],
        ),
    ]

    result = validate_dataset(str(csv_file), rules)

    assert result["total_rules"] == 3
    assert result["passed_rules"] == 3
    assert result["failed_rules"] == 0
    assert result["all_passed"] is True


def test_validate_dataset_detects_missing_required_column(tmp_path):
    csv_file = tmp_path / "missing_column.csv"
    pd.DataFrame({"age": [25, 35]}).to_csv(csv_file, index=False)

    rules = [
        RequiredColumnRule(
            type="required_column",
            column="name",
        )
    ]

    result = validate_dataset(str(csv_file), rules)

    assert result["all_passed"] is False
    assert result["results"][0]["passed"] is False
    assert "missing" in result["results"][0]["message"]


def test_validate_dataset_detects_numeric_range_violations(tmp_path):
    csv_file = tmp_path / "range.csv"
    pd.DataFrame({"age": [18, 25, 70, "unknown"]}).to_csv(
        csv_file, index=False
    )

    rules = [
        NumericRangeRule(
            type="numeric_range",
            column="age",
            minimum=18,
            maximum=65,
        )
    ]

    result = validate_dataset(str(csv_file), rules)

    assert result["failed_rules"] == 1
    assert result["results"][0]["invalid_rows"] == 2


def test_validate_dataset_detects_disallowed_values(tmp_path):
    csv_file = tmp_path / "allowed.csv"
    pd.DataFrame({
        "status": ["active", "inactive", "pending"]
    }).to_csv(csv_file, index=False)

    rules = [
        AllowedValuesRule(
            type="allowed_values",
            column="status",
            values=["active", "inactive"],
        )
    ]

    result = validate_dataset(str(csv_file), rules)

    assert result["all_passed"] is False
    assert result["results"][0]["invalid_rows"] == 1


def test_validate_dataset_reports_missing_column_for_value_rules(tmp_path):
    csv_file = tmp_path / "missing.csv"
    pd.DataFrame({"name": ["Alice"]}).to_csv(csv_file, index=False)

    rules = [
        NumericRangeRule(
            type="numeric_range",
            column="age",
            minimum=0,
            maximum=120,
        ),
        AllowedValuesRule(
            type="allowed_values",
            column="status",
            values=["active"],
        ),
    ]

    result = validate_dataset(str(csv_file), rules)

    assert result["failed_rules"] == 2
    assert all(not item["passed"] for item in result["results"])


def test_validate_dataset_handles_empty_csv(tmp_path):
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="no columns"):
        validate_dataset(str(csv_file), [])


def test_validate_dataset_handles_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_dataset(str(tmp_path / "missing.csv"), [])


# Dataset validation API tests

def test_dataset_validation_endpoint_success(tmp_path):
    csv_file = tmp_path / "valid.csv"
    pd.DataFrame({"name": ["Alice"], "age": [25]}).to_csv(
        csv_file, index=False
    )

    dataset = SimpleNamespace(
        id=1,
        filename="valid.csv",
        file_path=str(csv_file),
    )

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = (
        dataset
    )

    payload = {
        "rules": [
            {"type": "required_column", "column": "name"},
            {
                "type": "numeric_range",
                "column": "age",
                "minimum": 18,
                "maximum": 65,
            },
        ]
    }

    with patch.object(
        datasets_routes,
        "SessionLocal",
        return_value=mock_db,
    ):
        response = TestClient(app).post(
            "/datasets/1/validate",
            json=payload,
        )

    assert response.status_code == 200
    body = response.json()
    assert body["dataset_id"] == 1
    assert body["filename"] == "valid.csv"
    assert body["validation"]["all_passed"] is True
    assert body["validation"]["passed_rules"] == 2
    mock_db.close.assert_called_once()


def test_dataset_validation_endpoint_dataset_not_found():
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = None

    with patch.object(
        datasets_routes,
        "SessionLocal",
        return_value=mock_db,
    ):
        response = TestClient(app).post(
            "/datasets/999/validate",
            json={
                "rules": [
                    {
                        "type": "required_column",
                        "column": "name",
                    }
                ]
            },
        )

    assert response.status_code == 404
    assert response.json()["detail"] == "Dataset not found"
    mock_db.close.assert_called_once()


def test_dataset_validation_endpoint_file_not_found(tmp_path):
    dataset = SimpleNamespace(
        id=1,
        filename="missing.csv",
        file_path=str(tmp_path / "missing.csv"),
    )

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = (
        dataset
    )

    with patch.object(
        datasets_routes,
        "SessionLocal",
        return_value=mock_db,
    ):
        response = TestClient(app).post(
            "/datasets/1/validate",
            json={
                "rules": [
                    {
                        "type": "required_column",
                        "column": "name",
                    }
                ]
            },
        )

    assert response.status_code == 404
    assert response.json()["detail"] == "Dataset file not found"
    mock_db.close.assert_called_once()


def test_dataset_validation_endpoint_rejects_invalid_rules():
    client = TestClient(app)

    response = client.post(
        "/datasets/1/validate",
        json={"rules": [{"type": "unknown_rule", "column": "name"}]},
    )

    assert response.status_code == 422


def test_dataset_validation_endpoint_closes_db_on_unexpected_error(
    tmp_path,
):
    dataset = SimpleNamespace(
        id=1,
        filename="valid.csv",
        file_path=str(tmp_path / "valid.csv"),
    )

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = (
        dataset
    )

    with (
        patch.object(
            datasets_routes,
            "SessionLocal",
            return_value=mock_db,
        ),
        patch.object(
            datasets_routes,
            "validate_dataset",
            side_effect=RuntimeError("Unexpected failure"),
        ),
        pytest.raises(RuntimeError, match="Unexpected failure"),
    ):
        TestClient(app).post(
            "/datasets/1/validate",
            json={
                "rules": [
                    {
                        "type": "required_column",
                        "column": "name",
                    }
                ]
            },
        )

    mock_db.close.assert_called_once()

def test_list_dataset_versions_success():
    dataset = SimpleNamespace(id=101)

    version = SimpleNamespace(
        id=1,
        version_number=1,
        file_path="data/raw/sample.csv",
        row_count=3,
        column_count=2,
        created_at=None,
    )

    mock_db = MagicMock()

    dataset_query = MagicMock()
    dataset_query.filter.return_value.first.return_value = dataset

    versions_query = MagicMock()
    versions_query.filter.return_value.order_by.return_value.all.return_value = [
        version
    ]

    mock_db.query.side_effect = [dataset_query, versions_query]

    with patch.object(
        datasets_routes,
        "SessionLocal",
        return_value=mock_db,
    ):
        response = TestClient(app).get("/datasets/101/versions")

    assert response.status_code == 200
    assert response.json() == {
        "dataset_id": 101,
        "versions": [
            {
                "id": 1,
                "version_number": 1,
                "file_path": "data/raw/sample.csv",
                "row_count": 3,
                "column_count": 2,
                "created_at": None,
            }
        ],
    }

    mock_db.close.assert_called_once()


def test_list_dataset_versions_dataset_not_found():
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = None

    with patch.object(
        datasets_routes,
        "SessionLocal",
        return_value=mock_db,
    ):
        response = TestClient(app).get("/datasets/999999/versions")

    assert response.status_code == 404
    assert response.json() == {"detail": "Dataset not found"}

    mock_db.close.assert_called_once()


def test_list_dataset_versions_empty():
    dataset = SimpleNamespace(id=101)

    mock_db = MagicMock()

    dataset_query = MagicMock()
    dataset_query.filter.return_value.first.return_value = dataset

    versions_query = MagicMock()
    versions_query.filter.return_value.order_by.return_value.all.return_value = []

    mock_db.query.side_effect = [dataset_query, versions_query]

    with patch.object(
        datasets_routes,
        "SessionLocal",
        return_value=mock_db,
    ):
        response = TestClient(app).get("/datasets/101/versions")

    assert response.status_code == 200
    assert response.json() == {
        "dataset_id": 101,
        "versions": [],
    }

    mock_db.close.assert_called_once()

def test_compare_dataset_versions_success():
    old_version = SimpleNamespace(
        version_number=1,
        file_path="version_1.csv",
    )

    new_version = SimpleNamespace(
        version_number=2,
        file_path="version_2.csv",
    )

    mock_db = MagicMock()

    dataset_query = MagicMock()
    dataset_query.filter.return_value.first.return_value = (
        SimpleNamespace(id=201)
    )

    versions_query = MagicMock()
    versions_query.filter.return_value.all.return_value = [
        old_version,
        new_version,
    ]

    mock_db.query.side_effect = [
        dataset_query,
        versions_query,
    ]

    old_df = pd.DataFrame(
        {
            "name": ["Ali", None],
            "age": [25, 30],
            "city": ["Lahore", "Karachi"],
        }
    )

    new_df = pd.DataFrame(
        {
            "name": ["Ali", "Sara", None],
            "age": [25.0, None, 30.0],
            "salary": [50000, 60000, 70000],
        }
    )

    with (
        patch.object(
            datasets_routes,
            "SessionLocal",
            return_value=mock_db,
        ),
        patch.object(
            datasets_routes.pd,
            "read_csv",
            side_effect=[old_df, new_df],
        ),
    ):
        response = TestClient(app).get(
            "/datasets/201/versions/compare?from=1&to=2"
        )

    assert response.status_code == 200

    result = response.json()

    assert result["dataset_id"] == 201
    assert result["from_version"] == 1
    assert result["to_version"] == 2

    assert result["row_count"] == {
        "from": 2,
        "to": 3,
        "delta": 1,
    }

    assert result["column_count"] == {
        "from": 3,
        "to": 3,
        "delta": 0,
    }

    assert result["added_columns"] == ["salary"]
    assert result["removed_columns"] == ["city"]

    assert result["dtype_changes"] == [
        {
            "column": "age",
            "from": "int64",
            "to": "float64",
        }
    ]

    assert result["missing_value_changes"] == [
        {
            "column": "name",
            "from": 1,
            "to": 1,
            "delta": 0,
        },
        {
            "column": "age",
            "from": 0,
            "to": 1,
            "delta": 1,
        },
        {
            "column": "city",
            "from": 0,
            "to": None,
            "delta": None,
        },
        {
            "column": "salary",
            "from": None,
            "to": 0,
            "delta": None,
        },
    ]

    mock_db.close.assert_called_once()


def test_compare_dataset_versions_dataset_not_found():
    mock_db = MagicMock()

    mock_db.query.return_value.filter.return_value.first.return_value = (
        None
    )

    with patch.object(
        datasets_routes,
        "SessionLocal",
        return_value=mock_db,
    ):
        response = TestClient(app).get(
            "/datasets/999/versions/compare?from=1&to=2"
        )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Dataset not found"
    }

    mock_db.close.assert_called_once()


def test_compare_dataset_versions_version_not_found():
    mock_db = MagicMock()

    dataset_query = MagicMock()
    dataset_query.filter.return_value.first.return_value = (
        SimpleNamespace(id=201)
    )

    versions_query = MagicMock()
    versions_query.filter.return_value.all.return_value = [
        SimpleNamespace(
            version_number=1,
            file_path="version_1.csv",
        )
    ]

    mock_db.query.side_effect = [
        dataset_query,
        versions_query,
    ]

    with patch.object(
        datasets_routes,
        "SessionLocal",
        return_value=mock_db,
    ):
        response = TestClient(app).get(
            "/datasets/201/versions/compare?from=1&to=2"
        )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "One or both dataset versions were not found"
    }

    mock_db.close.assert_called_once()


def test_compare_dataset_versions_same_version():
    with patch.object(
        datasets_routes,
        "SessionLocal",
    ) as mock_session:
        response = TestClient(app).get(
            "/datasets/201/versions/compare?from=1&to=1"
        )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Cannot compare a version with itself"
    }

    mock_session.assert_not_called()


def test_compare_dataset_versions_invalid_version_parameters():
    with patch.object(
        datasets_routes,
        "SessionLocal",
    ) as mock_session:
        response = TestClient(app).get(
            "/datasets/201/versions/compare?from=0&to=2"
        )

    assert response.status_code == 422
    mock_session.assert_not_called()


def test_compare_dataset_versions_file_not_found():
    old_version = SimpleNamespace(
        version_number=1,
        file_path="missing_version.csv",
    )

    new_version = SimpleNamespace(
        version_number=2,
        file_path="version_2.csv",
    )

    mock_db = MagicMock()

    dataset_query = MagicMock()
    dataset_query.filter.return_value.first.return_value = (
        SimpleNamespace(id=201)
    )

    versions_query = MagicMock()
    versions_query.filter.return_value.all.return_value = [
        old_version,
        new_version,
    ]

    mock_db.query.side_effect = [
        dataset_query,
        versions_query,
    ]

    with (
        patch.object(
            datasets_routes,
            "SessionLocal",
            return_value=mock_db,
        ),
        patch.object(
            datasets_routes.pd,
            "read_csv",
            side_effect=FileNotFoundError,
        ),
    ):
        response = TestClient(app).get(
            "/datasets/201/versions/compare?from=1&to=2"
        )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Version file not found"
    }

    mock_db.close.assert_called_once()
