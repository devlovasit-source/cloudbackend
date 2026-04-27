from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.settings import settings

logger = logging.getLogger("ahvi.main")

# Ensure the repo root is importable when running `python main.py`
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_router(module_path: str, attr: str = "router"):
    """
    Import a router module and return its `router` object.
    Fail-soft so one broken router doesn't prevent the API from starting.
    """
    try:
        module = __import__(module_path, fromlist=[attr])
        return getattr(module, attr)
    except Exception as exc:  # noqa: BLE001
        logger.exception("router load failed module=%s error=%s", module_path, exc)
        return None


app = FastAPI(title="AHVI Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=(
        settings.cors_allow_credentials and not ("*" in settings.cors_allowed_origins)
    ),
    allow_methods=settings.cors_allowed_methods,
    allow_headers=settings.cors_allowed_headers,
)


# -------------------------
# ROUTERS
# -------------------------
_ROUTERS: list[tuple[str, dict[str, Any]]] = [
    ("routers.chat", {"prefix": "/api", "tags": ["Chat"]}),
    ("routers.data", {}),
    ("routers.utilities", {}),
    ("routers.boards", {}),
    ("routers.feedback", {}),
    ("routers.ops", {"prefix": "/api/ops", "tags": ["Ops"]}),
    ("routers.calendar", {"prefix": "/api"}),
    ("routers.notifications", {}),
    ("routers.stylist", {"prefix": "/api/stylist"}),
    ("routers.reddit", {}),
    ("routers.vision", {"prefix": "/api/vision"}),
    ("routers.wardrobe_capture", {}),
    ("routers.bg_router", {}),
    ("routers.garment_analyzer", {"prefix": "/api"}),
    ("api.ai", {"prefix": "/api", "tags": ["AI"]}),
]

ROUTER_STATUS: dict[str, str] = {}
for module_path, kwargs in _ROUTERS:
    router = _load_router(module_path)
    if router is None:
        ROUTER_STATUS[module_path] = "failed"
        continue
    app.include_router(router, **kwargs)
    ROUTER_STATUS[module_path] = "loaded"


# -------------------------
# HEALTH
# -------------------------
@app.get("/health")
def health():
    return {
        "status": "online",
        "ready": True,
        "routers": ROUTER_STATUS,
    }


@app.get("/")
def root():
    return {"message": "backend running"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

