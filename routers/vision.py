import base64
import os
from collections import Counter

import cv2
import numpy as np
import requests

RUNPOD_URL = "https://wvntzm71uikrla-11434.proxy.runpod.net/remove-bg"

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans

from services.embedding_service import encode_metadata
from services.image_embedding_service import encode_image_base64
from services.image_fingerprint import compute_pixel_hash_from_base64
from services.qdrant_service import qdrant_service
from services.task_queue import enqueue_task

try:
    from worker import vision_analyze_task
except Exception:
    vision_analyze_task = None


router = APIRouter()


class ImageAnalyzeRequest(BaseModel):
    image_base64: str = Field(..., min_length=20)
    userId: str = "demo_user"


# =========================
# BASE64 HELPERS
# =========================
def _normalize_base64_for_model(value: str) -> str:
    text = (value or "").strip()
    return text.split(",", 1)[1] if "," in text else text


def _to_png_data_uri(base64_text: str) -> str:
    text = _normalize_base64_for_model(base64_text)
    return f"data:image/png;base64,{text}"


def _input_has_alpha(image_base64: str) -> bool:
    try:
        b64 = _normalize_base64_for_model(image_base64)
        img_data = base64.b64decode(b64)
        np_arr = np.frombuffer(img_data, np.uint8)
        decoded = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        return bool(decoded is not None and decoded.ndim == 3 and decoded.shape[2] == 4)
    except Exception:
        return False


# =========================
# BG REMOVE
# =========================
def _remove_bg_first(image_base64: str):
    if _input_has_alpha(image_base64):
        return image_base64, True, "input_already_has_alpha"

    try:
        response = requests.post(
            RUNPOD_URL,
            json={"image_base64": image_base64},
            timeout=20
        )

        if response.status_code == 200:
            data = response.json()
            if data.get("success") and data.get("image_base64"):
                processed = _to_png_data_uri(data["image_base64"])
                return processed, True, None

    except Exception as e:
        print("[BG ERROR]", str(e))

    return image_base64, False, "runpod_failed"


# =========================
# COLOR
# =========================
def get_dominant_color(cv_image):
    try:
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (100, 100))
        pixels = image.reshape((-1, 3))

        kmeans = KMeans(n_clusters=3, n_init=10).fit(pixels)
        dominant = kmeans.cluster_centers_[Counter(kmeans.labels_).most_common(1)[0][0]]

        r, g, b = [int(x) for x in dominant]
        return "#{:02x}{:02x}{:02x}".format(r, g, b).upper()
    except Exception:
        return "#000000"


def _hex_to_color_name(hex_color: str) -> str:
    try:
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
    except:
        return "Multicolor"

    if max(r, g, b) < 40:
        return "Black"
    if min(r, g, b) > 220:
        return "White"
    if g > r and g > b:
        return "Green"
    if b > r and b > g:
        return "Blue"
    if r > g and r > b:
        return "Red"

    return "Multicolor"


# =========================
# SHAPE CLASSIFIER
# =========================
def _infer_garment_hint(decoded_img):
    h, w = decoded_img.shape[:2]
    ratio = h / w

    if ratio > 1.8:
        return "Bottoms", "Trousers"
    if ratio > 1.2:
        return "Tops", "Shirt"
    return "Tops", "T-Shirt"


def _infer_pattern(cv_image):
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return "checked" if np.mean(edges) > 10 else "plain"


# =========================
# CORE OUTPUT
# =========================
def _build_output(color_hex, decoded, cv_image):
    category, sub_category = _infer_garment_hint(decoded)

    # consistency fix
    if sub_category.lower() in ["shirt", "t-shirt"]:
        category = "Tops"
    if sub_category.lower() in ["trousers", "jeans", "pants"]:
        category = "Bottoms"

    color_name = _hex_to_color_name(color_hex)

    return {
        "name": f"{color_name} {sub_category}",
        "category": category,
        "sub_category": sub_category,
        "pattern": _infer_pattern(cv_image),
        "occasions": ["casual outing", "daily wear", "weekend", "travel", "office"],
        "color_code": color_hex,
    }


# =========================
# MAIN CORE
# =========================
def vision_analyze_core(image_base64: str, user_id: str = "demo_user"):
    image_base64, bg_removed, bg_reason = _remove_bg_first(image_base64)

    base64_data = _normalize_base64_for_model(image_base64)

    img_data = base64.b64decode(base64_data)
    np_arr = np.frombuffer(img_data, np.uint8)
    decoded = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if decoded is None:
        raise HTTPException(status_code=400, detail="invalid image")

    cv_image = decoded
    color_hex = get_dominant_color(cv_image)

    final_data = _build_output(color_hex, decoded, cv_image)
    final_data["userId"] = user_id

    return {
        "success": True,
        "data": final_data,
        "processed_image_base64": image_base64,
        "similar_items": [],
        "meta": {
            "bg_removed": bg_removed,
            "bg_fallback_reason": bg_reason,
            "llm_fallback": True,
            "vision_model_used": None,
        },
    }


# =========================
# ROUTE
# =========================
@router.post("/analyze-image")
def analyze_image(request: ImageAnalyzeRequest):
    return vision_analyze_core(request.image_base64, request.userId)


# =========================
# ASYNC (UNCHANGED)
# =========================
@router.post("/analyze-image/async", status_code=status.HTTP_202_ACCEPTED)
def analyze_image_async(http_request: Request, request: ImageAnalyzeRequest):
    if vision_analyze_task is None:
        raise HTTPException(status_code=503, detail="Worker not configured")

    task_id = enqueue_task(
        task_func=vision_analyze_task,
        args=[request.image_base64, request.userId],
        kwargs={"request_id": str(getattr(http_request.state, "request_id", "") or "")},
        kind="vision_analyze",
        user_id=request.userId,
    )

    return {"success": True, "status": "queued", "task_id": task_id}
