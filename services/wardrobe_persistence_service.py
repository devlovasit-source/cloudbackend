import os
import uuid
import base64
import requests
from typing import List, Dict, Any

from services.r2_storage import R2Storage
from services.qdrant_service import qdrant_service
from services.embedding_service import embedding_service


# =========================
# ENV CONFIG
# =========================
APPWRITE_ENDPOINT = os.getenv("APPWRITE_ENDPOINT")
APPWRITE_PROJECT_ID = os.getenv("APPWRITE_PROJECT_ID")
APPWRITE_API_KEY = os.getenv("APPWRITE_API_KEY")
APPWRITE_DATABASE_ID = os.getenv("APPWRITE_DATABASE_ID")
APPWRITE_COLLECTION_ID = os.getenv("APPWRITE_COLLECTION_ID")

HEADERS = {
    "X-Appwrite-Project": APPWRITE_PROJECT_ID,
    "X-Appwrite-Key": APPWRITE_API_KEY,
    "Content-Type": "application/json",
}

r2 = R2Storage()


# =========================
# HELPERS
# =========================
def _decode_base64(value: str) -> bytes:
    text = (value or "").strip()
    if not text:
        return b""
    if "," in text:
        text = text.split(",", 1)[1]
    try:
        return base64.b64decode(text)
    except Exception:
        return b""


def _create_document(document_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{APPWRITE_ENDPOINT}/databases/{APPWRITE_DATABASE_ID}/collections/{APPWRITE_COLLECTION_ID}/documents"

    payload = {
        "documentId": document_id,  # ✅ SAME ID
        "data": data,
    }

    res = requests.post(url, json=payload, headers=HEADERS)

    if res.status_code not in (200, 201):
        raise Exception(f"Appwrite error: {res.text}")

    return res.json()


# =========================
# CATEGORY NORMALIZATION
# =========================
CATEGORY_MAP = {
    "tops": "top",
    "top": "top",
    "bottoms": "bottom",
    "bottom": "bottom",
    "footwear": "footwear",
    "shoes": "footwear",
}


def normalize_category(cat: str) -> str:
    if not cat:
        return "top"
    return CATEGORY_MAP.get(cat.lower(), "top")


# =========================
# MAIN FUNCTION
# =========================
def persist_selected_items(
    user_id: str,
    selected_item_ids: List[str],
    detected_items: List[Dict[str, Any]],
):
    """
    FINAL PIPELINE:
    - Upload to R2
    - Save to Appwrite
    - Save to Qdrant
    """

    saved_items = []

    for item in detected_items:
        if item.get("item_id") not in selected_item_ids:
            continue

        try:
            # =========================
            # 🔥 USE SAME ID EVERYWHERE
            # =========================
            file_id = item.get("item_id") or str(uuid.uuid4())

            # -------------------------
            # 1. Decode images
            # -------------------------
            raw_bytes = _decode_base64(item.get("raw_crop_base64", ""))
            mask_bytes = _decode_base64(item.get("segmented_png_base64", ""))

            if not raw_bytes or not mask_bytes:
                print("⚠️ Skipping item due to missing image data")
                continue

            # -------------------------
            # 2. Upload to R2
            # -------------------------
            upload_result = r2.upload_wardrobe_images(
                file_id=file_id,
                raw_image_bytes=raw_bytes,
                masked_image_bytes=mask_bytes,
            )

            raw_url = upload_result.get("raw_image_url", "")
            mask_url = upload_result.get("masked_image_url", "")

            if not raw_url or not mask_url:
                print("⚠️ R2 upload failed")
                continue

            # -------------------------
            # 3. Normalize fields
            # -------------------------
            category = normalize_category(item.get("category"))
            item_type = str(item.get("sub_category", "")).lower()
            color = item.get("color_code", "#000000")

            # -------------------------
            # 4. Build embedding
            # -------------------------
            embedding = embedding_service.encode_text(
                f"{color} {item_type} {category}"
            )

            # -------------------------
            # 5. Appwrite document
            # -------------------------
            doc = {
                "userId": user_id,
                "status": "active",

                "image_url": raw_url,
                "masked_url": mask_url,

                "image_id": file_id,
                "masked_id": file_id,

                "qdrant_point_id": file_id,

                "name": item.get("name", "Item"),
                "category": category,
                "sub_category": item_type,
                "color_code": color,
                "pattern": item.get("pattern", "plain"),
                "occasions": item.get("occasions", []),

                "worn": 0,
                "liked": False,
            }

            created = _create_document(file_id, doc)

            # -------------------------
            # 🔥 6. SAVE TO QDRANT (CRITICAL)
            # -------------------------
            try:
                qdrant_service.upsert_wardrobe_item({
                    "id": file_id,
                    "userId": user_id,

                    "type": item_type,
                    "category": category,
                    "color": color,

                    "image_url": mask_url,

                    "embedding": embedding,
                })

                print("✅ QDRANT SAVED")

            except Exception as e:
                print("❌ QDRANT ERROR:", str(e))

            saved_items.append(created)

        except Exception as e:
            print("❌ Persist error:", str(e))

    return {
        "success": True,
        "saved_count": len(saved_items),
        "items": saved_items,
    }
