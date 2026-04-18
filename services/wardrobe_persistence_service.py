import os
import uuid
import base64
import requests
from typing import List, Dict, Any

from services.r2_storage import R2Storage


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


def _create_document(data: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{APPWRITE_ENDPOINT}/databases/{APPWRITE_DATABASE_ID}/collections/{APPWRITE_COLLECTION_ID}/documents"

    payload = {
        "documentId": str(uuid.uuid4()),
        "data": data,
    }

    res = requests.post(url, json=payload, headers=HEADERS)

    if res.status_code not in (200, 201):
        raise Exception(f"Appwrite error: {res.text}")

    return res.json()


# =========================
# MAIN FUNCTION
# =========================
def persist_selected_items(
    user_id: str,
    selected_item_ids: List[str],
    detected_items: List[Dict[str, Any]],
):
    """
    Full pipeline:
    - Decode base64 images
    - Upload to R2
    - Store metadata in Appwrite
    """

    saved_items = []

    for item in detected_items:
        if item.get("item_id") not in selected_item_ids:
            continue

        try:
            # -------------------------
            # 1. Decode images
            # -------------------------
            raw_bytes = _decode_base64(item.get("raw_crop_base64", ""))
            mask_bytes = _decode_base64(item.get("segmented_png_base64", ""))

            if not raw_bytes or not mask_bytes:
                print("⚠️ Skipping item due to missing image data")
                continue

            # -------------------------
            # 2. Upload to R2 (CORRECT)
            # -------------------------
            file_id = str(uuid.uuid4())

            upload_result = r2.upload_wardrobe_images(
                file_id=file_id,
                raw_image_bytes=raw_bytes,
                masked_image_bytes=mask_bytes,
            )

            raw_url = upload_result.get("raw_image_url", "")
            mask_url = upload_result.get("masked_image_url", "")

            raw_id = upload_result.get("raw_file_name", file_id)
            mask_id = upload_result.get("masked_file_name", file_id)

            if not raw_url or not mask_url:
                print("⚠️ R2 upload failed, skipping item")
                continue

            # -------------------------
            # 3. Prepare Appwrite doc
            # -------------------------
            doc = {
                # REQUIRED
                "userId": user_id,
                "status": "active",

                "image_url": raw_url,
                "masked_url": mask_url,

                "image_id": raw_id,
                "masked_id": mask_id,

                "qdrant_point_id": str(uuid.uuid4()),

                # METADATA
                "name": item.get("name", "Item"),
                "category": item.get("category", "Tops"),
                "sub_category": item.get("sub_category", "item"),
                "color_code": item.get("color_code", "#000000"),
                "pattern": item.get("pattern", "plain"),
                "occasions": item.get("occasions", []),

                # USAGE
                "worn": 0,
                "liked": False,
            }

            created = _create_document(doc)
            saved_items.append(created)

        except Exception as e:
            print("❌ Persist error:", str(e))

    return {
        "success": True,
        "saved_count": len(saved_items),
        "items": saved_items,
    }
