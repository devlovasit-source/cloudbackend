import base64
import io
import uuid
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from PIL import Image

from services.hybrid_detection_service import run_hybrid_detection
from services.wardrobe_persistence_service import persist_selected_items
from services.image_embedding_service import encode_image_base64

router = APIRouter(prefix="/api/wardrobe/capture", tags=["wardrobe-capture"])


# =========================
# REQUEST MODELS
# =========================
class CaptureAnalyzeRequest(BaseModel):
    user_id: str
    image_base64: str = Field(..., min_length=20)
    auto_save: bool = False


class SaveSelectedRequest(BaseModel):
    user_id: str
    selected_item_ids: List[str]
    detected_items: List[Dict[str, Any]]


# =========================
# HELPERS
# =========================
def _decode_image_base64(value: str) -> Image.Image:
    text = (value or "").strip()
    if "," in text:
        text = text.split(",", 1)[1]

    try:
        data = base64.b64decode(text, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image_base64: {exc}")

    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (max 15MB)")

    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image bytes: {exc}")


# =========================
# MAIN ANALYZE (UPDATED)
# =========================
@router.post("/analyze")
async def analyze_capture(http_request: Request, request: CaptureAnalyzeRequest):

    # -------------------------
    # DECODE IMAGE
    # -------------------------
    image = _decode_image_base64(request.image_base64)

    # -------------------------
    # 🔥 HYBRID DETECTION (NEW)
    # -------------------------
    try:
        detected_items = await run_hybrid_detection(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {e}")

    items = []

    # -------------------------
    # POST-PROCESS ITEMS
    # -------------------------
    for item in detected_items:

        try:
            # embedding (temporary base64 for embedding only)
            embedding = encode_image_base64(item["raw_url"]) if item.get("raw_url") else []
        except Exception:
            embedding = []

        items.append({
            "item_id": item["item_id"],
            "name": item.get("label", "Item"),

            # 🔥 IMPORTANT MAPPING
            "category": item.get("label", "Tops"),
            "sub_category": item.get("label", "item"),

            "color_code": "#000000",  # can improve later
            "pattern": "plain",
            "occasions": [],

            "confidence": 0.8,
            "reasoning": "hybrid_detection",

            # 🔥 NEW PIPELINE (URL BASED)
            "raw_url": item.get("raw_url"),
            "masked_url": item.get("masked_url"),

            "image_embedding": embedding,
        })

    # -------------------------
    # FALLBACK
    # -------------------------
    if not items:
        items = [{
            "item_id": str(uuid.uuid4()),
            "name": "Fallback Item",
            "category": "Tops",
            "sub_category": "item",
            "color_code": "#000000",
            "pattern": "plain",
            "occasions": ["casual"],
            "confidence": 0.3,
            "reasoning": "fallback_no_detection",
            "raw_url": None,
            "masked_url": None,
            "image_embedding": [],
        }]

    # -------------------------
    # AUTO SAVE (UNCHANGED)
    # -------------------------
    if bool(request.auto_save):
        try:
            persist_selected_items(
                user_id=request.user_id,
                selected_item_ids=[i["item_id"] for i in items],
                detected_items=items,
            )
        except Exception as exc:
            print("[wardrobe.capture] auto_save failed:", str(exc))

    return {
        "success": True,
        "items": items,
        "count": len(items),
    }


# =========================
# SAVE ENDPOINT (UNCHANGED)
# =========================
@router.post("/save-selected")
def save_selected(request: SaveSelectedRequest):
    return persist_selected_items(
        user_id=request.user_id,
        selected_item_ids=request.selected_item_ids,
        detected_items=request.detected_items,
    )
