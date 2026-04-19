
import base64
import io
import uuid
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from PIL import Image

from prompts.core_prompts import WARDROBE_CAPTURE_PROMPT
from services.wardrobe_persistence_service import persist_selected_items
from services import ai_gateway
from services.image_embedding_service import encode_image_base64

router = APIRouter(prefix="/api/wardrobe/capture", tags=["wardrobe-capture"])


# =========================
# REQUEST MODELS
# =========================
class CaptureAnalyzeRequest(BaseModel):
    user_id: str
    image_base64: str = Field(..., min_length=20)


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


def _image_to_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _dominant_hex(crop: Image.Image) -> str:
    arr = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
    arr = cv2.resize(arr, (100, 100))
    pixels = arr.reshape((-1, 3))

    if len(pixels) == 0:
        return "#000000"

    mean = np.mean(pixels, axis=0).astype(int)
    rgb = (int(mean[2]), int(mean[1]), int(mean[0]))
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _segment_png_base64(crop: Image.Image) -> str:
    arr = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
    h, w = arr.shape[:2]

    if h < 10 or w < 10:
        rgba = cv2.cvtColor(arr, cv2.COLOR_BGR2BGRA)
        return base64.b64encode(cv2.imencode(".png", rgba)[1].tobytes()).decode("utf-8")

    mask = np.zeros(arr.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    rect = (2, 2, max(1, w - 4), max(1, h - 4))

    try:
        cv2.grabCut(arr, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        alpha = np.where((mask == 2) | (mask == 0), 0, 255).astype("uint8")
    except Exception:
        alpha = np.full((h, w), 255, dtype=np.uint8)

    rgba = cv2.cvtColor(arr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha

    ok, encoded = cv2.imencode(".png", rgba)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode segmented PNG")

    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def _safe_bbox(raw_bbox: Any, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(raw_bbox, dict):
        return None

    try:
        x1 = max(0, min(width - 1, int(raw_bbox.get("x1", 0))))
        y1 = max(0, min(height - 1, int(raw_bbox.get("y1", 0))))
        x2 = max(1, min(width, int(raw_bbox.get("x2", width))))
        y2 = max(1, min(height, int(raw_bbox.get("y2", height))))
    except Exception:
        return None

    if x2 <= x1 or y2 <= y1:
        return None

    if (x2 - x1) < 24 or (y2 - y1) < 24:
        return None

    return (x1, y1, x2, y2)


# =========================
# MAIN ANALYZE
# =========================
@router.post("/analyze")
def analyze_capture(http_request: Request, request: CaptureAnalyzeRequest):

    request_id = str(getattr(http_request.state, "request_id", "") or "")
    image = _decode_image_base64(request.image_base64)
    width, height = image.size

    llm_items = []

    try:
        llm_items = ai_gateway.ollama_vision_json(
            prompt=WARDROBE_CAPTURE_PROMPT,
            image_base64=request.image_base64,
            request_id=request_id,
            usecase="vision",
        )[0].get("items", [])
    except Exception:
        pass

    items = []

    for row in llm_items:
        bbox_tuple = _safe_bbox(row.get("bbox"), width, height)
        if not bbox_tuple:
            continue

        x1, y1, x2, y2 = bbox_tuple
        crop = image.crop((x1, y1, x2, y2))

        raw_crop_base64 = _image_to_base64(crop)
        segmented_png_base64 = _segment_png_base64(crop)

        # embedding
        try:
            embedding = encode_image_base64(raw_crop_base64)
        except Exception:
            embedding = []

        items.append({
            "item_id": str(uuid.uuid4()),
            "name": row.get("name", "Item"),
            "category": row.get("category", "Tops"),
            "sub_category": row.get("sub_category", "item"),
            "color_code": row.get("color_code", _dominant_hex(crop)),
            "pattern": row.get("pattern", "plain"),
            "occasions": row.get("occasions", []),
            "confidence": row.get("confidence", 0.5),
            "reasoning": row.get("reasoning", ""),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "raw_crop_base64": raw_crop_base64,
            "segmented_png_base64": segmented_png_base64,
            "image_embedding": embedding,
        })

    # fallback
    if not items:
        crop = image
        raw_crop_base64 = _image_to_base64(crop)
        segmented_png_base64 = _segment_png_base64(crop)

        try:
            embedding = encode_image_base64(raw_crop_base64)
        except Exception:
            embedding = []

        items = [{
            "item_id": str(uuid.uuid4()),
            "name": "Fallback Item",
            "category": "Tops",
            "sub_category": "item",
            "color_code": _dominant_hex(crop),
            "pattern": "plain",
            "occasions": ["casual"],
            "confidence": 0.3,
            "reasoning": "Fallback detection",
            "bbox": {"x1": 0, "y1": 0, "x2": width, "y2": height},
            "raw_crop_base64": raw_crop_base64,
            "segmented_png_base64": segmented_png_base64,
            "image_embedding": embedding,
        }]

    # =========================
    # 🔥 AUTO SAVE (DEMO SAFE)
    # =========================
    try:
        persist_selected_items(
            user_id=request.user_id,
            selected_item_ids=[i["item_id"] for i in items],
            detected_items=items,
        )
        print("AUTO SAVE SUCCESS ✅")
    except Exception as e:
        print("AUTO SAVE FAILED ❌", e)

    return {
        "success": True,
        "items": items,
        "count": len(items),
    }


# =========================
# OPTIONAL SAVE ENDPOINT
# =========================
@router.post("/save-selected")
def save_selected(request: SaveSelectedRequest):
    return persist_selected_items(
        user_id=request.user_id,
        selected_item_ids=request.selected_item_ids,
        detected_items=request.detected_items,
    )
