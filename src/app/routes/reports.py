import logging
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.app import metrics
from src.app.config import get_settings
from src.app.database import get_db
from src.app.models.dataset_version import DatasetVersion
from src.app.models.ml_run import MLRun
from src.app.models.report import Report
from src.app.routes.common import get_dataset_or_404, load_df_or_http
from src.app.schemas.reports import ReportRequest
from src.app.services import cleaning as cleaning_service
from src.app.services import eda as eda_service
from src.app.services.ai.errors import AIServiceError
from src.app.services.ai.explain import build_facts, explain
from src.app.services.ai.provider import get_llm_optional
from src.app.services.data_io import sha256_file
from src.app.services.ml.service import run_to_dict
from src.app.services.reporting.builder import build_report_pdf

logger = logging.getLogger(__name__)
router = APIRouter(tags=["reports"])
FATAL_AI = {"auth", "not_configured", "quota", "rate_limited", "model_unavailable", "busy"}


def report_to_dict(report: Report) -> dict:
    return {"id": report.id, "dataset_id": report.dataset_id, "title": report.title,
            "size_bytes": report.size_bytes, "include_ai": report.include_ai,
            "ai_included": report.ai_included, "sections": report.sections,
            "created_at": report.created_at.isoformat() if report.created_at else None,
            "download_url": f"/reports/{report.id}/download"}


def _select_runs(db: Session, dataset_id: int, run_ids: list[int] | None) -> list[MLRun]:
    query = db.query(MLRun).filter(MLRun.dataset_id == dataset_id, MLRun.status == "completed")
    if run_ids:
        runs = query.filter(MLRun.id.in_(run_ids)).all()
        if len(runs) != len(set(run_ids)):
            raise HTTPException(status_code=404, detail="One or more model runs were not found")
        return runs
    best: dict[str, MLRun] = {}
    for run in query.all():
        current = best.get(run.task)
        if current is None or (run.primary_metric_value or -1e18) > (current.primary_metric_value or -1e18):
            best[run.task] = run
    return list(best.values())


@router.post("/datasets/{dataset_id}/reports")
def create_report(dataset_id: int, body: ReportRequest, db: Session = Depends(get_db),
                  llm_pair=Depends(get_llm_optional)):
    dataset = get_dataset_or_404(db, dataset_id)
    df = load_df_or_http(dataset)
    settings = get_settings()
    settings.ensure_dirs()

    eda = eda_service.generate_eda(df)
    recs = cleaning_service.analyze_cleaning(df)["recommendations"]
    runs = [run_to_dict(r) for r in _select_runs(db, dataset_id, body.run_ids)]

    ai_sections: list[dict] = []
    ai_notes: list[str] = []
    if body.include_ai:
        llm, unavailable = llm_pair
        if llm is None:
            ai_notes.append(f"AI explanations were requested but unavailable: {unavailable}")
        else:
            jobs = [("insights", "Key insights and recommendations", None),
                    ("quality", "Data quality explained", None)]
            jobs += [("ml_run", f"Model explained: {r['model_name']} v{r['version']}", r)
                     for r in runs[:3]]
            for kind, label, run in jobs:
                try:
                    facts = build_facts(kind, df, dataset.filename, run=run)
                    result = explain(llm, kind, facts)
                    ai_sections.append({"title": label, "text": result["explanation"],
                                        "grounding": result["grounding"],
                                        "model": result["model"]})
                except AIServiceError as exc:
                    ai_notes.append(f"'{label}' could not be generated: {exc.message}")
                    if exc.kind in FATAL_AI:
                        break

    version = (db.query(func.max(DatasetVersion.version_number))
               .filter(DatasetVersion.dataset_id == dataset_id).scalar())
    path = Path(dataset.file_path)
    title = body.title or f"Data Intelligence Report - {Path(dataset.filename).stem}"
    out_path = settings.reports_dir / f"{uuid4()}_{Path(dataset.filename).stem[:50]}_report.pdf"
    try:
        info = build_report_pdf(
            out_path, title=title,
            dataset={"id": dataset.id, "filename": dataset.filename, "version": version,
                     "sha256": sha256_file(path) if path.is_file() else None,
                     "created_at": dataset.created_at.isoformat()[:19] if dataset.created_at else None},
            df=df, eda=eda, cleaning_recs=recs, runs=runs, ai_sections=ai_sections,
            ai_notes=ai_notes, include_ai=body.include_ai)
    except Exception:
        out_path.unlink(missing_ok=True)
        logger.exception("report generation failed")
        raise HTTPException(status_code=500, detail="Report generation failed") from None

    report = Report(dataset_id=dataset_id, title=title, file_path=str(out_path),
                    size_bytes=out_path.stat().st_size, include_ai=body.include_ai,
                    ai_included=bool(ai_sections),
                    sections={"sections": info["sections"], "charts": info["charts"],
                              "models": len(runs), "ai_notes": ai_notes})
    db.add(report)
    db.commit()
    db.refresh(report)
    metrics.incr("reports_generated")
    return report_to_dict(report)


@router.get("/datasets/{dataset_id}/reports")
def list_reports(dataset_id: int, db: Session = Depends(get_db)):
    get_dataset_or_404(db, dataset_id)
    reports = (db.query(Report).filter(Report.dataset_id == dataset_id)
               .order_by(Report.id.desc()).all())
    return [report_to_dict(r) for r in reports]


def _get_report(db: Session, report_id: int) -> Report:
    report = db.query(Report).filter(Report.id == report_id).first()
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return report


@router.get("/reports/{report_id}/download")
def download_report(report_id: int, db: Session = Depends(get_db)):
    report = _get_report(db, report_id)
    path = Path(report.file_path).resolve()
    if get_settings().reports_dir.resolve() not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="Report file not found")
    return FileResponse(path, media_type="application/pdf",
                        filename=f"{Path(report.title).stem[:80] or 'report'}.pdf")


@router.delete("/reports/{report_id}")
def delete_report(report_id: int, db: Session = Depends(get_db)):
    report = _get_report(db, report_id)
    path = Path(report.file_path)
    db.delete(report)
    db.commit()
    path.unlink(missing_ok=True)
    return {"message": "Report deleted", "id": report_id}
