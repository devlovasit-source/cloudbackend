import base64
import io
import os
import asyncio
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from PIL import Image
import requests

from services.detection_pipeline import run_hybrid_detection
from services.embedding_service import encode_metadata
from services.image_embedding_service import encode_image_base64
from services.qdrant_service import qdrant_service
from services import ai_gateway

router = APIRouter()

# =========================
# CONFIG
# =========================
HF_API_KEY = os.getenv("HF_API_KEY")
HF_BG_MODEL = "briaai/RMBG-2.0"

# =========================
# REQUEST
# =========================
class ImageAnalyzeRequest(BaseModel):
    image_base64: str = Field(..., min_length=20)
    userId: str = "demo_user"

# =========================
# HELPERS
# =========================
def _normalize_base64(value: str) -> str:
    return value.split(",", 1)[1] if "," in value else value

def _decode_pil_image(image_base64: str) -> Image.Image:
    try:
        base64_data = _normalize_base64(image_base64)
        image_bytes = base64.b64decode(base64_data)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

# =========================
# 🔥 HF BG REMOVAL (sync, offloaded later)
# =========================
def _remove_bg_hf_sync(image_base64: str):
    if not HF_API_KEY:
        return image_base64, False, "no_api_key"

    try:
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        img_bytes = base64.b64decode(_normalize_base64(image_base64))

        res = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_BG_MODEL}",
            headers=headers,
            data=img_bytes,
            timeout=20,
        )

        if res.status_code != 200:
            return image_base64, False, "hf_failed"

        return base64.b64encode(res.content).decode(), True, None

    except Exception as e:
        print("[vision] bg removal error:", e)
        return image_base64, False, "exception"

# =========================
# 🔥 PROCESS SINGLE ITEM (PARALLEL UNIT)
# =========================
async def _process_single_item(item: dict, user_id: str):
    try:
        item_id = item.get("item_id")
        masked_url = item.get("masked_url")

        final_data = {
            "category": item.get("label"),
            "image_url": item.get("raw_url"),
            "masked_url": masked_url,
            "userId": user_id,
            "image_id": item_id,
        }

        loop = asyncio.get_event_loop()

        # =========================
        # 🔥 AI ENRICHMENT (NON-BLOCKING)
        # =========================
        ai_data = {}

        try:
            # Skip heavy AI for tiny accessories
            if final_data["category"] not in ["ring", "earring", "bracelet"]:
                ai_data, _ = await loop.run_in_executor(
                    None,
                    lambda: ai_gateway.ollama_vision_json(
                        prompt="Describe fashion attributes",
                        image_base64=masked_url,
                        usecase="vision",
                    ),
                )
        except Exception as e:
            print("[vision] AI error:", e)

        final_data.update({
            "pattern": ai_data.get("pattern"),
            "occasions": ai_data.get("occasions"),
            "style": ai_data.get("style"),
        })

        # =========================
        # 🔥 PARALLEL EMBEDDINGS
        # =========================
        text_future = loop.run_in_executor(None, encode_metadata, final_data)
        image_future = loop.run_in_executor(None, encode_image_base64, masked_url)

        text_vector, image_vector = await asyncio.gather(text_future, image_future)

        if text_vector and image_vector:
            vector = [(0.6 * iv + 0.4 * tv) for iv, tv in zip(image_vector, text_vector)]
        else:
            vector = image_vector or text_vector or []

        # =========================
        # 🔍 SEARCH
        # =========================
        similar_items = []
        if vector:
            similar_items = qdrant_service.search_similar(vector, user_id, limit=5)

        # =========================
        # 💾 SAVE (NON-BLOCKING)
        # =========================
        if vector:
            await loop.run_in_executor(
                None,
                lambda: qdrant_service.upsert_item(
                    item_id=item_id,
                    vector=vector,
                    payload=final_data,
                ),
            )

        return {
            "item_id": item_id,
            "category": final_data.get("category"),
            "image_url": final_data.get("image_url"),
            "masked_url": final_data.get("masked_url"),
            "similar_items": similar_items,
        }

    except Exception as e:
        print("[vision] item failed:", e)
        return None

# =========================
# 🔥 MAIN PIPELINE
# =========================
async def process_items(image: Image.Image, image_base64: str, user_id: str):

    loop = asyncio.get_event_loop()

    # BG removal (offloaded)
    processed_base64, bg_removed, bg_error = await loop.run_in_executor(
        None, lambda: _remove_bg_hf_sync(image_base64)
    )

    # Detection (your service)
    detections = await run_hybrid_detection(image)

    if not detections:
        return {
            "items": [],
            "meta": {
                "bg_removed": bg_removed,
                "bg_error": bg_error,
            },
        }

    # =========================
    # 🔥 PARALLEL ITEM PROCESSING
    # =========================
    tasks = [_process_single_item(item, user_id) for item in detections]
    results = await asyncio.gather(*tasks)

    results = [r for r in results if r]

    return {
        "items": results,
        "meta": {
            "bg_removed": bg_removed,
            "bg_error": bg_error,
        },
    }

# =========================
# ROUTE
# =========================
@router.post("/analyze-image")
async def analyze_image(request: ImageAnalyzeRequest):

    image = _decode_pil_image(request.image_base64)

    result = await process_items(image, request.image_base64, request.userId)

    return {
        "success": True,
        "total_items": len(result["items"]),
        "items": result["items"],
        "meta": result["meta"],
    }
