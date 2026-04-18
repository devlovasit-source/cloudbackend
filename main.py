
import asyncio
import logging
import importlib.util
import os
from uuid import uuid4
from time import perf_counter
from typing import Callable, Any, Dict

from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# SERVICES
from services.qdrant_service import qdrant_service
from services.security_limits import (
    check_rate_limit,
    extract_client_ip,
    is_redis_rate_limit_ready,
)
from services.settings import settings
from middleware.auth_middleware import get_current_user
from services.job_tracker import job_tracker
from services.request_context import set_request_id

logger = logging.getLogger("ahvi.main")

# -------------------------
# ROUTER LOADER
# -------------------------
def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _load_optional_router(module_name: str, attr: str = "router"):
    if not _has_module(module_name):
        logger.info("router skipped module=%s reason=not_found", module_name)
        return None
    try:
        module = __import__(module_name, fromlist=[attr])
        return getattr(module, attr)
    except Exception as exc:
        logger.warning(
            "router load failed module=%s error=%s type=%s",
            module_name,
            str(exc),
            type(exc).__name__,
        )
        return None


# -------------------------
# LOAD ROUTERS
# -------------------------
chat_router = _load_optional_router("routers.chat")
data_router = _load_optional_router("routers.data")
utilities_router = _load_optional_router("routers.utilities")
boards_router = _load_optional_router("routers.boards")
feedback_router = _load_optional_router("routers.feedback")
ops_router = _load_optional_router("routers.ops")
ai_router = _load_optional_router("api.ai")

# -------------------------
# APP INIT
# -------------------------
app = FastAPI(
    title="AHVI AI Master Brain API",
    version="3.0.0",
)

logger.info("AHVI Backend Started")

# -------------------------
# STARTUP / SHUTDOWN
# -------------------------
_qdrant_initialized = False


@app.on_event("startup")
async def startup_event():
    global _qdrant_initialized

    start_time = perf_counter()
    logger.info("startup begin")

    if _qdrant_initialized:
        logger.info("qdrant already initialized, skipping")
        return

    try:
        logger.info("initializing qdrant...")
        await asyncio.to_thread(qdrant_service.init)
        _qdrant_initialized = True
        logger.info("qdrant initialized successfully")
    except Exception as e:
        logger.exception("qdrant startup failed: %s", e)

    logger.info("startup complete in %sms", int((perf_counter() - start_time) * 1000))


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("shutdown begin")

    try:
        client = getattr(qdrant_service, "client", None)
        if client and hasattr(client, "close"):
            await asyncio.to_thread(client.close)
    except Exception as e:
        logger.warning("qdrant shutdown error=%s", e)


# -------------------------
# ERROR HANDLERS
# -------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"success": False, "error": "Invalid request", "details": exc.errors()},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error"},
    )


# -------------------------
# MIDDLEWARE
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_tracing_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    set_request_id(request_id)
    request.state.request_id = request_id

    # 🔥 SYSTEM READY CHECK
    if not hasattr(qdrant_service, "client"):
        return JSONResponse(
            status_code=503,
            content={"success": False, "error": "System not ready"},
        )

    start = perf_counter()
    response = await call_next(request)
    latency = int((perf_counter() - start) * 1000)

    response.headers["X-Request-ID"] = request_id

    logger.info(
        "request id=%s method=%s path=%s latency=%sms",
        request_id,
        request.method,
        request.url.path,
        latency,
    )

    return response


# -------------------------
# ROUTERS
# -------------------------
if chat_router:
    app.include_router(chat_router, prefix="/api")

if data_router:
    app.include_router(data_router)

if utilities_router:
    app.include_router(utilities_router)

if boards_router:
    app.include_router(boards_router)

if ai_router:
    app.include_router(ai_router, prefix="/api")

if feedback_router:
    app.include_router(feedback_router)

if ops_router:
    app.include_router(ops_router)


# -------------------------
# HEALTH / DEBUG
# -------------------------
@app.get("/")
def root():
    return {"message": "AHVI backend running"}


@app.get("/health")
def health():
    return {
        "status": "online",
        "service": "ahvi-brain",
        "version": "3.0.0",
    }


@app.get("/ready")
def ready():
    return {
        "status": "ready" if hasattr(qdrant_service, "client") else "not_ready"
    }


@app.get("/debug/system")
def debug():
    return {
        "routers": {
            "chat": bool(chat_router),
            "ai": bool(ai_router),
        },
        "qdrant_ready": hasattr(qdrant_service, "client"),
    }


# -------------------------
# BASIC JOB STATUS
# -------------------------
@app.get("/api/jobs/recent")
def list_jobs():
    return {
        "success": True,
        "jobs": job_tracker.list_recent(limit=20),
    }
