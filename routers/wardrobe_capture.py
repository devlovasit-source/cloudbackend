import base64
import io
import uuid
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, status, Request
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw

from prompts.core_prompts import WARDROBE_CAPTURE_PROMPT

from services.wardrobe_persistence_service import persist_selected_items
from services import ai_gateway
from services import hf_qwen_service
from services.image_embedding_service import encode_image_base64  # 🔥 NEW

try:
    from worker import capture_analyze_task, capture_save_selected_task, process_upload_task
except Exception:
    capture_analyze_task = None
    capture_save_selected_task = None
    process_upload_task = None

try:
    from services.job_tracker import job_tracker
except Exception:
    job_tracker = None

from services.task_queue import enqueue_task


router = APIRouter(prefix="/api/wardrobe/capture", tags=["wardrobe-capture"])


class CaptureAnalyzeRequest(BaseModel):
    user_id: str
    image_base64: str = Field(..., min_length=20)


class DetectedItem(BaseModel):
    item_id: str
    name: str
    category: str
    sub_category: str
    color_code: str
    pattern: str = "plain"
    occasions: List[str] = Field(default_factory=lambda: ["casual"])
    confidence: float = 0.0
    reasoning: str
    bbox: Dict[str, int]
    raw_crop_base64: str
    segmented_png_base64: str
    image_embedding: List[float] = []  # 🔥 NEW


class SaveSelectedRequest(BaseModel):
    user_id: str
    selected_item_ids: List[str]
    detected_items: List[DetectedItem]


class ProcessUploadRequest(BaseModel):
    user_id: str
    image_base64: str = Field(..., min_length=20)


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
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image bytes: {exc}")

    return image


def _image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _dominant_hex(crop: Image.Image) -> str:
    arr = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
    arr = cv2.resize(arr, (100, 100), interpolation=cv2.INTER_AREA)
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
        x1 = max(0, min(width - 1, int(float(raw_bbox.get("x1", 0)))))
        y1 = max(0, min(height - 1, int(float(raw_bbox.get("y1", 0)))))
        x2 = max(1, min(width, int(float(raw_bbox.get("x2", width)))))
        y2 = max(1, min(height, int(float(raw_bbox.get("y2", height)))))
    except Exception:
        return None

    if x2 <= x1:
        x2 = min(width, x1 + 2)
    if y2 <= y1:
        y2 = min(height, y1 + 2)
    if (x2 - x1) < 24 or (y2 - y1) < 24:
        return None

    return (x1, y1, x2, y2)


def _fallback_single_item(image: Image.Image) -> List[Dict[str, Any]]:
    width, height = image.size
    crop = image.crop((0, 0, width, height))

    raw_crop_base64 = _image_to_base64(crop, fmt="JPEG")

    # 🔥 embedding
    image_embedding = []
    try:
        image_embedding = encode_image_base64(raw_crop_base64)
        if not image_embedding or len(image_embedding) < 100:
            image_embedding = []
    except Exception:
        image_embedding = []

    return [
        {
            "item_id": str(uuid.uuid4()),
            "name": "Detected Outfit Item",
            "category": "Tops",
            "sub_category": "item",
            "color_code": _dominant_hex(crop),
            "pattern": "plain",
            "occasions": ["casual"],
            "confidence": 0.35,
            "reasoning": "Fallback item generated.",
            "bbox": {"x1": 0, "y1": 0, "x2": width, "y2": height},
            "raw_crop_base64": raw_crop_base64,
            "segmented_png_base64": _segment_png_base64(crop),
            "image_embedding": image_embedding,
        }
    ]


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

        raw_crop_base64 = _image_to_base64(crop, fmt="JPEG")
        segmented_png_base64 = _segment_png_base64(crop)

        # 🔥 embedding
        image_embedding = []
        try:
            image_embedding = encode_image_base64(raw_crop_base64)
            if not image_embedding or len(image_embedding) < 100:
                image_embedding = []
        except Exception:
            image_embedding = []

        items.append(
            {
                "item_id": str(uuid.uuid4()),
                "name": row.get("name", "Item"),
                "category": row.get("category", ""),
                "sub_category": row.get("sub_category", ""),
                "color_code": row.get("color_code", _dominant_hex(crop)),
                "pattern": row.get("pattern", "plain"),
                "occasions": row.get("occasions", []),
                "confidence": row.get("confidence", 0.5),
                "reasoning": row.get("reasoning", ""),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "raw_crop_base64": raw_crop_base64,
                "segmented_png_base64": segmented_png_base64,
                "image_embedding": image_embedding,
            }
        )

    if not items:
        items = _fallback_single_item(image)

    return {
        "success": True,
        "items": items,
        "count": len(items),
    }
