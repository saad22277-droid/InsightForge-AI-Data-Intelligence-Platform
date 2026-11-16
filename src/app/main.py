import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.app import metrics
from src.app.config import get_settings
from src.app.logging_config import setup_logging
from src.app.routes.ai import router as ai_router
from src.app.routes.analysis import router as analysis_router
from src.app.routes.cleaning import router as cleaning_router
from src.app.routes.datasets import router as datasets_router
from src.app.routes.experiments import router as experiments_router
from src.app.routes.ml import router as ml_router
from src.app.routes.reports import router as reports_router
from src.app.routes.system import router as system_router
from src.app.services.ai.errors import AIServiceError

settings = get_settings()
setup_logging(settings.log_level, settings.log_json)
settings.ensure_dirs()
logger = logging.getLogger("app.http")


app = FastAPI(
    title="AI Data Intelligence Platform",
    description="AI, Data Science and MLOps platform",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    """Structured access log + metrics. Logs route and status only - never bodies."""
    request_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        metrics.incr("http_5xx")
        logger.exception("unhandled error", extra={"request_id": request_id,
                                                   "path": request.url.path})
        raise
    elapsed = time.perf_counter() - started
    route = request.scope.get("route")
    template = getattr(route, "path", request.url.path)
    metrics.incr("http_requests")
    metrics.observe("http", elapsed)
    if response.status_code >= 500:
        metrics.incr("http_5xx")
    elif response.status_code >= 400:
        metrics.incr("http_4xx")
    if request.url.path not in {"/health", "/metrics"}:
        logger.info("request", extra={
            "request_id": request_id, "method": request.method, "route": template,
            "status": response.status_code, "duration_ms": round(elapsed * 1000, 1)})
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(AIServiceError)
async def ai_error_handler(request: Request, exc: AIServiceError):
    metrics.incr("ai_errors")
    headers = {"Retry-After": str(int(exc.retry_after))} if exc.retry_after else None
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": {"code": exc.kind, "message": exc.message}},
        headers=headers,
    )


@app.get("/")
def root():
    return {
        "message": "AI Data Platform is running",
        "version": "0.1.0",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


app.include_router(datasets_router)
app.include_router(analysis_router)
app.include_router(cleaning_router)
app.include_router(ml_router)
app.include_router(experiments_router)
app.include_router(ai_router)
app.include_router(reports_router)
app.include_router(system_router)
