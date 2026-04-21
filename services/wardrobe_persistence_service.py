import base64
import os
import re
import uuid
from typing import Any, Dict, List

import requests

from services.embedding_service import embedding_service
from services.qdrant_service import qdrant_service
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
_HEX6_RE = re.compile(r"^[0-9a-fA-F]{6}$")


def _appwrite_ready() -> bool:
    return bool(
        APPWRITE_ENDPOINT
        and APPWRITE_PROJECT_ID
        and APPWRITE_API_KEY
        and APPWRITE_DATABASE_ID
        and APPWRITE_COLLECTION_ID
    )


def _normalize_hex_color(value: str, default: str = "#000000") -> str:
    """
    Normalize a color into canonical '#RRGGBB' uppercase.
    Accepts '#RGB', '#RRGGBB', 'RRGGBB', 'RGB'. Falls back to default.
    """
    text = (value or "").strip()
    if not text:
        return default
    if text.startswith("#"):
        text = text[1:]
    if len(text) == 3 and re.match(r"^[0-9a-fA-F]{3}$", text):
        text = "".join([c * 2 for c in text])
    if not _HEX6_RE.match(text):
        return default
    return f"#{text.upper()}"


def _decode_base64(value: str) -> bytes:
    text = (value or "").strip()
    if not text:
        return b""
    if "," in text:
        text = text.split(",", 1)[1]
    try:
        # validate=False keeps backward-compat with non-strict base64 payloads.
        return base64.b64decode(text, validate=False)
    except Exception:
        return b""


def _create_document(document_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    if not _appwrite_ready():
        raise Exception("Appwrite not configured")

    url = f"{APPWRITE_ENDPOINT}/databases/{APPWRITE_DATABASE_ID}/collections/{APPWRITE_COLLECTION_ID}/documents"
    payload = {"documentId": document_id, "data": data}

    res = requests.post(url, json=payload, headers=HEADERS, timeout=20)
    if res.status_code not in (200, 201):
        raise Exception(f"Appwrite error: {res.text}")
    return res.json()


# =========================
# CATEGORY NORMALIZATION
# =========================
CATEGORY_MAP = {
    # Canonical categories match what the UI and vision pipeline use.
    "tops": "Tops",
    "top": "Tops",
    "t": "Tops",
    "bottoms": "Bottoms",
    "bottom": "Bottoms",
    "b": "Bottoms",
    "outerwear": "Outerwear",
    "jacket": "Outerwear",
    "jackets": "Outerwear",
    "dresses": "Dresses",
    "dress": "Dresses",
    "footwear": "Footwear",
    "shoes": "Footwear",
    "sneakers": "Footwear",
    "bags": "Bags",
    "bag": "Bags",
    "accessories": "Accessories",
    "accessory": "Accessories",
    "jewelry": "Jewelry",
    "jewellery": "Jewelry",
    "indian wear": "Indian Wear",
    "ethnic": "Indian Wear",
}


def normalize_category(cat: str) -> str:
    if not cat:
        return "Tops"
    return CATEGORY_MAP.get(str(cat).strip().lower(), "Tops")


# =========================
# MAIN FUNCTION
# =========================
def persist_selected_items(
    user_id: str,
    selected_item_ids: List[str],
    detected_items: List[Dict[str, Any]],
):
    """
    Pipeline:
    - Upload to R2 (raw + masked PNG)
    - Save document to Appwrite (if configured)
    - Upsert vector + metadata to Qdrant (best-effort)
    """

    saved_items: List[Dict[str, Any]] = []
    skipped_missing_data = 0
    skipped_upload_failed = 0
    persist_failures = 0

    selected_ids = set([str(x) for x in (selected_item_ids or [])])

    for item in (detected_items or []):
        if str(item.get("item_id") or "") not in selected_ids:
            continue

        try:
            file_id = str(item.get("item_id") or "").strip() or str(uuid.uuid4())

            raw_bytes = _decode_base64(item.get("raw_crop_base64", ""))
            mask_bytes = _decode_base64(item.get("segmented_png_base64", ""))
            if not raw_bytes or not mask_bytes:
                skipped_missing_data += 1
                continue

            upload_result = r2.upload_wardrobe_images(
                file_id=file_id,
                raw_image_bytes=raw_bytes,
                masked_image_bytes=mask_bytes,
            )
            raw_url = upload_result.get("raw_image_url", "")
            mask_url = upload_result.get("masked_image_url", "")
            if not raw_url or not mask_url:
                skipped_upload_failed += 1
                continue

            category = normalize_category(item.get("category"))
            sub_category = str(item.get("sub_category", "") or "Item").strip()
            item_type_slug = sub_category.lower()
            color = _normalize_hex_color(item.get("color_code", "#000000"))
            pattern = str(item.get("pattern") or "plain").strip().lower() or "plain"
            occasions = item.get("occasions", [])
            if not isinstance(occasions, list):
                occasions = []

            embedding = embedding_service.encode_text(f"{color} {item_type_slug} {category} {pattern}")

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
                "sub_category": sub_category,
                "color_code": color,
                "pattern": pattern,
                "occasions": occasions,
                "worn": 0,
                "liked": False,
            }

            if _appwrite_ready():
                created = _create_document(file_id, doc)
            else:
                # Dev fallback when Appwrite isn't configured.
                created = {"$id": file_id, "data": doc}

            try:
                qdrant_service.upsert_wardrobe_item(
                    {
                        "id": file_id,
                        "userId": user_id,
                        "type": item_type_slug,
                        "category": category,
                        "color": color,
                        "image_url": mask_url,
                        "embedding": embedding,
                    }
                )
            except Exception as e:
                # Keep persistence usable even if vector store is down.
                print("[wardrobe.persist] qdrant error:", str(e))

            saved_items.append(created)

        except Exception as e:
            persist_failures += 1
            print("[wardrobe.persist] persist error:", str(e))

    if not saved_items:
        return {
            "success": False,
            "saved_count": 0,
            "items": [],
            "error": "no_items_saved",
            "skipped_missing_data": skipped_missing_data,
            "skipped_upload_failed": skipped_upload_failed,
            "persist_failures": persist_failures,
        }

    return {
        "success": True,
        "saved_count": len(saved_items),
        "items": saved_items,
        "skipped_missing_data": skipped_missing_data,
        "skipped_upload_failed": skipped_upload_failed,
        "persist_failures": persist_failures,
    }

