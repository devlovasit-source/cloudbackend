import base64
import os
import time
import hashlib
import uuid
import requests
from collections import Counter

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans

from services import ai_gateway
from services.embedding_service import encode_metadata
from services.image_embedding_service import encode_image_base64
from services.image_fingerprint import compute_pixel_hash_from_base64
from services.qdrant_service import qdrant_service
from services.task_queue import enqueue_task
from brain.engines.color_normalizer import color_normalizer
from services.wardrobe_persistence_service import persist_selected_items

try:
    from worker import vision_analyze_task
except Exception:
    vision_analyze_task = None

try:
    from routers.bg_remover import BGRemoveRequest, remove_background_sync
except Exception:
    BGRemoveRequest = None
    remove_background_sync = None

router = APIRouter()


# =========================
# ENV HELPERS
# =========================
def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, str(default))).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _vision_enable_similarity() -> bool:
    return _env_bool("VISION_ANALYZE_ENABLE_SIMILARITY", False)


def _enable_local_rembg_fallback() -> bool:
    # Keep false by default so chat pods do not run heavy local segmentation.
    return _env_bool("ENABLE_LOCAL_REMBG_FALLBACK", False)


def _duplicate_threshold() -> float:
    try:
        val = float(os.getenv("WARDROBE_DUPLICATE_THRESHOLD", "0.97"))
        return val if 0.0 < val <= 1.0 else 0.97
    except Exception:
        return 0.97


def _pixel_duplicate_distance() -> int:
    try:
        val = int(os.getenv("WARDROBE_PIXEL_DUPLICATE_DISTANCE", "6"))
        return max(0, min(val, 64))
    except Exception:
        return 6


def _image_duplicate_threshold() -> float:
    try:
        val = float(os.getenv("WARDROBE_IMAGE_DUPLICATE_THRESHOLD", "0.985"))
        return val if 0.0 < val <= 1.0 else 0.985
    except Exception:
        return 0.985


def _vision_max_image_bytes() -> int:
    try:
        val = int(os.getenv("VISION_MAX_IMAGE_BYTES", str(12 * 1024 * 1024)))
        return max(512 * 1024, val)
    except Exception:
        return 12 * 1024 * 1024


class ImageAnalyzeRequest(BaseModel):
    image_base64: str = Field(..., min_length=20)
    userId: str = "demo_user"
    auto_save: bool = False


# =========================
# BASE64 HELPERS
# =========================
def _normalize_base64_for_model(value: str) -> str:
    text = (value or "").strip()
    return text.split(",", 1)[1] if "," in text else text


def _to_png_data_uri(base64_text: str) -> str:
    text = _normalize_base64_for_model(base64_text)
    return f"data:image/png;base64,{text}"


def _ensure_png_base64(image_base64: str) -> str:
    b64 = _normalize_base64_for_model(image_base64)
    img_data = base64.b64decode(b64, validate=True)
    np_arr = np.frombuffer(img_data, np.uint8)
    decoded = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise HTTPException(status_code=400, detail="unable to decode image for PNG conversion")
    ok, encoded = cv2.imencode(".png", decoded)
    if not ok:
        raise HTTPException(status_code=500, detail="unable to encode PNG")
    return base64.b64encode(encoded.tobytes()).decode()


def _decode_and_validate_image(image_base64: str, max_bytes: int):
    try:
        base64_data = _normalize_base64_for_model(image_base64)
        img_data = base64.b64decode(base64_data, validate=True)
        if len(img_data) > max_bytes:
            raise HTTPException(status_code=413, detail=f"image payload too large (max {max_bytes} bytes)")
        np_arr = np.frombuffer(img_data, np.uint8)
        decoded = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise HTTPException(status_code=400, detail="invalid image payload")

        cv_image = (
            cv2.cvtColor(decoded, cv2.COLOR_BGRA2BGR)
            if (decoded.ndim == 3 and decoded.shape[2] == 4)
            else (decoded if decoded.ndim == 3 else cv2.imdecode(np_arr, cv2.IMREAD_COLOR))
        )
        if cv_image is None:
            raise HTTPException(status_code=400, detail="invalid image payload")

        return base64_data, decoded, cv_image
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image payload: {str(e)}")


def _persist_vision_result(
    *,
    user_id: str,
    original_image_base64: str,
    vision_payload: dict,
) -> dict:
    if not isinstance(vision_payload, dict):
        raise HTTPException(status_code=500, detail="invalid vision payload for persistence")

    data = vision_payload.get("data") if isinstance(vision_payload.get("data"), dict) else {}
    meta = vision_payload.get("meta") if isinstance(vision_payload.get("meta"), dict) else {}
    processed = str(vision_payload.get("processed_image_base64") or "").strip()
    if not processed:
        raise HTTPException(status_code=502, detail="missing processed image for persistence")

    if not bool(meta.get("bg_removed")):
        raise HTTPException(status_code=409, detail="background not removed; cannot save masked image")

    raw_png_b64 = _ensure_png_base64(original_image_base64)
    masked_png_b64 = _ensure_png_base64(processed)

    item_id = str(uuid.uuid4())
    detected_item = {
        "item_id": item_id,
        "name": data.get("name") or "Item",
        "category": data.get("category") or "Tops",
        "sub_category": data.get("sub_category") or "item",
        "color_code": data.get("color_code") or "#000000",
        "pattern": data.get("pattern") or "plain",
        "occasions": data.get("occasions") or [],
        "raw_crop_base64": raw_png_b64,
        "segmented_png_base64": masked_png_b64,
    }

    return persist_selected_items(
        user_id=str(user_id or "demo_user"),
        selected_item_ids=[item_id],
        detected_items=[detected_item],
    )


def _input_has_alpha(image_base64: str) -> bool:
    try:
        b64 = _normalize_base64_for_model(image_base64)
        img_data = base64.b64decode(b64, validate=True)
        np_arr = np.frombuffer(img_data, np.uint8)
        decoded = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        return bool(decoded is not None and decoded.ndim == 3 and decoded.shape[2] == 4)
    except Exception:
        return False


# =========================
# BG OPTIMIZATION HELPERS
# =========================
_BG_CACHE = {}
_BG_CACHE_MAX_ITEMS = max(64, int(os.getenv("VISION_BG_CACHE_MAX_ITEMS", "512")))


def _hash_base64(b64: str) -> str:
    return hashlib.md5(b64.encode()).hexdigest()


def _bg_cache_get(cache_key: str):
    row = _BG_CACHE.get(cache_key)
    if not isinstance(row, dict):
        return None
    return row.get("value")


def _bg_cache_set(cache_key: str, value: str) -> None:
    _BG_CACHE[cache_key] = {"value": value, "time": time.time()}
    if len(_BG_CACHE) > _BG_CACHE_MAX_ITEMS:
        oldest_key = min(_BG_CACHE.items(), key=lambda kv: float(kv[1].get("time") or 0.0))[0]
        _BG_CACHE.pop(oldest_key, None)


def _resize_if_needed(base64_str: str, max_size=1024):
    try:
        img_bytes = base64.b64decode(base64_str)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return base64_str

        h, w = img.shape[:2]
        if max(h, w) <= max_size:
            return base64_str

        scale = max_size / max(h, w)
        resized = cv2.resize(img, (int(w * scale), int(h * scale)))

        _, buffer = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return base64.b64encode(buffer).decode()

    except Exception:
        return base64_str


# =========================
# BG REMOVE (OPTIMIZED)
# =========================
def _remove_bg_first(image_base64: str):
    base64_clean = _normalize_base64_for_model(image_base64)

    # 1. Skip if already alpha
    if _input_has_alpha(image_base64):
        return image_base64, True, "already_alpha"

    # 2. Resize for performance
    base64_clean = _resize_if_needed(base64_clean)

    # 3. Cache
    cache_key = _hash_base64(base64_clean)
    cached = _bg_cache_get(cache_key)
    if cached:
        return cached, True, "cache_hit"

    # 4. Local BG remover (Railway)
    if BGRemoveRequest is not None and remove_background_sync is not None:
        try:
            req = BGRemoveRequest(image_base64=base64_clean)
            result = remove_background_sync(req.image_base64)

            if isinstance(result, dict) and result.get("success") and result.get("image_base64"):
                processed = _to_png_data_uri(result.get("image_base64"))
                _bg_cache_set(cache_key, processed)
                return processed, True, "local_success"

        except Exception as exc:
            print(f"[BG] local failed: {exc}")

    # 5. RunPod (GPU)
    RUNPOD_URL = os.getenv("RUNPOD_BG_SINGLE_URL", "https://wvntzm71uikrla-11434.proxy.runpod.net/remove-bg")

    for attempt in range(2):
        try:
            response = requests.post(
                RUNPOD_URL,
                json={"image_base64": base64_clean},
                timeout=int(os.getenv("RUNPOD_BG_TIMEOUT_SECONDS", "10"))
            )

            if response.ok:
                data = response.json()
                if data.get("success") and data.get("image_base64"):
                    processed = _to_png_data_uri(data["image_base64"])
                    _bg_cache_set(cache_key, processed)
                    return processed, True, f"runpod_success_{attempt+1}"

        except requests.exceptions.Timeout:
            print(f"[BG] timeout attempt {attempt+1}")
        except Exception as e:
            print(f"[BG] error attempt {attempt+1}: {e}")

    # 6. Optional local rembg fallback (disabled by default for chat pods).
    if _enable_local_rembg_fallback():
        try:
            from rembg import remove

            img_bytes = base64.b64decode(base64_clean)
            output = remove(img_bytes)

            encoded = base64.b64encode(output).decode()
            processed = f"data:image/png;base64,{encoded}"

            _bg_cache_set(cache_key, processed)
            return processed, True, "rembg_fallback"

        except Exception as e:
            print("[BG] rembg failed:", e)

    # 7. Final fallback
    return image_base64, False, "all_failed"


# =========================
# COLOR
# =========================
def get_dominant_color(cv_image, k=3):
    try:
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        crop_h, crop_w = int(h * 0.25), int(w * 0.25)
        center_image = image[crop_h:h - crop_h, crop_w:w - crop_w]
        center_image = cv2.resize(center_image, (100, 100), interpolation=cv2.INTER_AREA)

        hsv_image = cv2.cvtColor(center_image, cv2.COLOR_RGB2HSV)
        pixels_rgb = center_image.reshape((-1, 3))
        pixels_hsv = hsv_image.reshape((-1, 3))

        mask = (pixels_hsv[:, 1] > 20) & (pixels_hsv[:, 2] > 70) & (pixels_hsv[:, 2] < 245)
        filtered_rgb = pixels_rgb[mask]
        if len(filtered_rgb) < 100:
            filtered_rgb = pixels_rgb[(pixels_hsv[:, 2] > 30) & (pixels_hsv[:, 2] < 250)]
            if len(filtered_rgb) == 0:
                filtered_rgb = pixels_rgb

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(filtered_rgb)
        dominant_rgb = [int(x) for x in kmeans.cluster_centers_[Counter(kmeans.labels_).most_common(1)[0][0]]]
        return "#{:02x}{:02x}{:02x}".format(*dominant_rgb).upper()
    except Exception:
        return "#000000"


def _hex_to_color_name(hex_color: str) -> str:
    try:
        color = hex_color.lstrip("#")
        r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
    except Exception:
        return "Multicolor"

    if max(r, g, b) < 40:
        return "Black"
    if min(r, g, b) > 220:
        return "White"
    if abs(r - g) < 14 and abs(g - b) < 14:
        return "Gray"
    if r > 180 and g < 110 and b < 110:
        return "Red"
    if r > 170 and g > 120 and b < 90:
        return "Orange"
    if r > 170 and g > 170 and b < 90:
        return "Yellow"
    if g > 150 and r < 130 and b < 130:
        return "Green"
    if b > 150 and r < 130 and g < 150:
        return "Blue"
    if r > 150 and b > 150 and g < 130:
        return "Purple"
    if r > 140 and g > 100 and b > 70:
        return "Brown"
    return "Multicolor"


# Emergency fallbacks used only when model output is missing fields.
def _extract_foreground_mask(decoded_img) -> np.ndarray | None:
    try:
        if decoded_img is None:
            return None
        if decoded_img.ndim == 3 and decoded_img.shape[2] == 4:
            return decoded_img[:, :, 3] > 18

        bgr = decoded_img if decoded_img.ndim == 3 else None
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        maxc, minc = rgb.max(axis=2), rgb.min(axis=2)
        near_white = (rgb[:, :, 0] >= 236) & (rgb[:, :, 1] >= 236) & (rgb[:, :, 2] >= 236) & ((maxc - minc) <= 20)
        saturated = hsv[:, :, 1] > 18
        mask = (~near_white) | saturated

        mask_u8 = mask.astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
        final_mask = mask_u8 > 0
        return final_mask if final_mask.sum() >= 200 else None
    except Exception:
        return None


def _infer_garment_hint(decoded_img) -> tuple[str, str]:
    mask = _extract_foreground_mask(decoded_img)
    if mask is None:
        return ("Tops", "Shirt")
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return ("Tops", "Shirt")
    box_w, box_h = max(1, xs.max() - xs.min() + 1), max(1, ys.max() - ys.min() + 1)
    if float(box_h) / float(box_w) > 1.15:
        return ("Bottoms", "Trousers")
    return ("Tops", "Shirt")


def _infer_pattern(cv_image) -> str:
    try:
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        return "checked" if (float(np.count_nonzero(edges)) / float(edges.size)) > 0.09 else "plain"
    except Exception:
        return "plain"


MASTER_VISION_PROMPT = """
You are a high-end fashion stylist vision classifier.
Analyze the garment image and return STRICT JSON with this exact shape:
{
  "name": "Highly descriptive name including the target gender if apparent (e.g., 'Men's Plain White Shirt', 'Women's Floral Midi Dress', 'Unisex Black Hoodie') if possible try to give in clour with sub category",
  "category": "Main category (Choose ONE: Tops, Bottoms, Dresses, Outerwear, Footwear, Bags, Accessories, Jewelry, Indian Wear)",
  "sub_category": "Specific type (e.g., T-Shirt, Chinos, Sneakers, Watch, Kurta)",
  "pattern": "one short value like plain/striped/checked/floral/graphic/printed/textured/denim",
  "occasions": ["list 5 to 8 specific occasions where this item can be worn"]
}

Rules:
- Accurately detect Footwear, Bags, and Accessories if applicable.
- Return EXACTLY 5 to 8 specific, highly creative occasions.
- Use lowercase strings for pattern and occasions.
"""


def _clean_text(val):
    return str(val).strip() if val else ""


def _normalize_occasions(raw_occ) -> list[str]:
    if isinstance(raw_occ, str):
        raw_occ = [x.strip() for x in raw_occ.split(",")]
    if not isinstance(raw_occ, list):
        return []
    out = []
    seen = set()
    for item in raw_occ:
        text = _clean_text(item).lower()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


_VALID_CATEGORIES = {
    "Tops",
    "Bottoms",
    "Footwear",
    "Outerwear",
    "Dresses",
    "Bags",
    "Accessories",
    "Jewelry",
    "Indian Wear",
}


def _shape_vision_output(raw_data, color_hex: str, decoded_img, cv_image) -> dict:
    data = dict(raw_data) if isinstance(raw_data, dict) else {}

    name = _clean_text(data.get("name") or data.get("title"))
    category = _clean_text(data.get("category") or data.get("main_category")).title()
    sub_category = _clean_text(data.get("sub_category") or data.get("subcategory") or data.get("subType")).title()
    pattern = _clean_text(data.get("pattern") or data.get("texture")).lower()
    occasions = _normalize_occasions(data.get("occasions") or data.get("occasion"))

    if not category or not sub_category:
        print("[vision] AI missing category/sub_category -> using emergency geometry fallback")
        fallback_cat, fallback_sub = _infer_garment_hint(decoded_img)
        category = category or fallback_cat
        sub_category = sub_category or fallback_sub

    if not name:
        name = f"{_hex_to_color_name(color_hex)} {sub_category}"

    if not pattern:
        print("[vision] AI missing pattern -> using emergency edge fallback")
        pattern = _infer_pattern(cv_image)

    if len(occasions) < 3:
        print("[vision] AI missing occasions -> using emergency generic fallback")
        occasions = ["daily wear", "casual outing", "weekend", "travel", "office", "hangout"]

    if category not in _VALID_CATEGORIES:
        category = "Tops"

    return {
        "name": name,
        "category": category,
        "sub_category": sub_category,
        "pattern": pattern,
        "occasions": occasions[:8],
        "color_code": color_hex,
    }

# =========================
# 🔥 ELITE INTELLIGENCE LAYER
# =========================

# =========================
# ELITE INTELLIGENCE LAYER
# =========================
def _build_items_from_single(final_data):
    return [{
        "type": str(final_data.get("sub_category") or "item").lower(),
        "color": final_data.get("color_code", "#000000"),
        "style": "casual",
    }]


def _analyze_outfit_relationship(items):
    types = [i.get("type", "") for i in items]
    colors = [i.get("color", "") for i in items]

    has_top = any(t in ["shirt", "t-shirt", "top"] for t in types)
    has_bottom = any(t in ["pants", "jeans", "trousers"] for t in types)

    completeness = "complete" if (has_top and has_bottom) else "partial"
    harmony = "clean" if len(set(colors)) <= 2 else "busy"

    styles = [i.get("style", "casual") for i in items]
    consistency = "cohesive" if len(set(styles)) == 1 else "mixed"

    return {
        "completeness": completeness,
        "color_harmony": harmony,
        "style_consistency": consistency,
    }


def _score_outfit(rel):
    score = 70
    if rel.get("completeness") == "complete":
        score += 10
    if rel.get("color_harmony") == "clean":
        score += 10
    if rel.get("style_consistency") == "cohesive":
        score += 10
    return min(score, 100)


def _generate_improvements(items, rel):
    suggestions = []

    if rel.get("completeness") == "partial":
        suggestions.append({
            "type": "add_item",
            "message": "Add a bottom to complete the outfit",
            "action": "add_bottom",
        })

    if rel.get("color_harmony") == "busy":
        suggestions.append({
            "type": "color_fix",
            "message": "Too many colors - simplify palette",
            "action": "simplify_colors",
        })

    if rel.get("style_consistency") == "mixed":
        suggestions.append({
            "type": "style_fix",
            "message": "Align styles for a cleaner look",
            "action": "align_style",
        })

    return suggestions


def _build_style_meta(item):
    color = str(item.get("color_code") or "#000000").upper()
    expression_tone = "minimal" if color in ["#000000", "#FFFFFF", "#888888"] else "expressive"
    color_tone = color_normalizer.detect_tone(color)
    return {
        "tone": expression_tone,
        "color_tone": color_tone,
        "versatility": "high" if item.get("pattern") == "plain" else "medium",
    }


def _build_visual_intelligence(final_data, items, rel, style_meta):
    color_hex = str(final_data.get("color_code") or "#000000")
    return {
        "dominant_color_hex": color_hex,
        "dominant_color_name": _hex_to_color_name(color_hex),
        "temperature_tone": style_meta.get("color_tone"),
        "expression_tone": style_meta.get("tone"),
        "item_type": str(final_data.get("sub_category") or ""),
        "style_consistency": rel.get("style_consistency"),
        "color_harmony": rel.get("color_harmony"),
        "completeness": rel.get("completeness"),
        "items_detected": len(items or []),
    }


@router.post("/analyze-image")
def analyze_image(request: ImageAnalyzeRequest):
    try:
        payload = vision_analyze_core(request.image_base64, request.userId)
        if request.auto_save:
            try:
                save_result = _persist_vision_result(
                    user_id=request.userId,
                    original_image_base64=request.image_base64,
                    vision_payload=payload,
                )
                payload["save"] = {
                    "enabled": True,
                    "success": True,
                    "result": save_result,
                }
            except HTTPException as exc:
                payload["save"] = {
                    "enabled": True,
                    "success": False,
                    "error": {"status_code": exc.status_code, "detail": str(exc.detail)},
                }
            except Exception as exc:
                payload["save"] = {
                    "enabled": True,
                    "success": False,
                    "error": {"status_code": 500, "detail": str(exc)},
                }
        return payload
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"vision analyze failed: {exc}")


def vision_analyze_core(image_base64: str, user_id: str = "demo_user"):
    max_bytes = _vision_max_image_bytes()
    # Validate payload first so invalid/oversized uploads fail before heavy BG processing.
    _, _, _ = _decode_and_validate_image(image_base64, max_bytes=max_bytes)

    vision_input_base64, bg_removed, bg_fallback_reason = _remove_bg_first(image_base64)
    base64_data, decoded, cv_image = _decode_and_validate_image(vision_input_base64, max_bytes=max_bytes)
    extracted_color_hex = get_dominant_color(cv_image)

    llm_fallback = False
    model_used = None
    try:
        final_data, model_used = ai_gateway.ollama_vision_json(
            prompt=MASTER_VISION_PROMPT,
            image_base64=base64_data,
            usecase="vision",
        )
    except Exception as e:
        print(f"[vision] AI Vision Error: {e}")
        llm_fallback = True
        final_data = {}

    final_data = _shape_vision_output(final_data, extracted_color_hex, decoded, cv_image)
    final_data["userId"] = user_id
    # =========================
    # ELITE INTELLIGENCE
    # =========================

    items = _build_items_from_single(final_data)
    rel = _analyze_outfit_relationship(items)
    score = _score_outfit(rel)
    improvements = _generate_improvements(items, rel)
    style_meta = _build_style_meta(final_data)
    visual_intelligence = _build_visual_intelligence(final_data, items, rel, style_meta)

    image_duplicate = {"checked": False, "is_duplicate": False, "id": None, "score": 0.0}
    pixel_duplicate = {"checked": False, "is_duplicate": False, "id": None, "distance": None}
    vector = None
    similar_items = []
    image_vector = []
    pixel_hash = ""
    image_duplicate_threshold = _image_duplicate_threshold()
    pixel_max_distance = _pixel_duplicate_distance()

    if _vision_enable_similarity():
        try:
            vector = encode_metadata(final_data)
            similar_items = qdrant_service.search_similar(vector, user_id, limit=5)
        except Exception as e:
            print(f"[vision] Similarity metadata search error: {e}")

        image_vector = encode_image_base64(vision_input_base64)
        if image_vector:
            try:
                image_duplicate = qdrant_service.find_image_duplicate(
                    image_vector, user_id, threshold=image_duplicate_threshold
                )
            except Exception as e:
                print(f"[vision] Image duplicate check error: {e}")

        pixel_hash = compute_pixel_hash_from_base64(vision_input_base64)
        if pixel_hash:
            try:
                pixel_duplicate = qdrant_service.find_pixel_duplicate(
                    user_id, pixel_hash, max_distance=pixel_max_distance
                )
            except Exception as e:
                print(f"[vision] Pixel duplicate check error: {e}")

    top_similarity_score = float(similar_items[0].get("score") or 0.0) if similar_items else 0.0
    probable_duplicate = bool(
        image_duplicate.get("is_duplicate")
        or pixel_duplicate.get("is_duplicate")
        or top_similarity_score >= _duplicate_threshold()
    )

    return {
        "success": True,
        "data": final_data,
        "items": items,
        "outfit": {
            "score": score,
            "analysis": rel,
        },
        "style": style_meta,
        "visual_intelligence": visual_intelligence,
        "improvements": improvements,
        "processed_image_base64": vision_input_base64,
        "similar_items": similar_items,
        "meta": {
            "bg_removed": bg_removed,
            "bg_fallback_reason": bg_fallback_reason,
            "llm_fallback": llm_fallback,
            "vision_model_used": model_used,
            "similarity_enabled": _vision_enable_similarity(),
            "embedding_created": vector is not None,
            "similar_items_found": len(similar_items),
            "image_duplicate_checked": bool(image_duplicate.get("checked")),
            "image_duplicate_threshold": image_duplicate_threshold,
            "duplicate_threshold": image_duplicate_threshold,
            "image_duplicate_score": float(image_duplicate.get("score") or 0.0),
            "image_duplicate_point_id": image_duplicate.get("id"),
            "pixel_duplicate_checked": bool(pixel_duplicate.get("checked")),
            "pixel_duplicate_distance": pixel_duplicate.get("distance"),
            "pixel_duplicate_max_distance": pixel_max_distance,
            "pixel_duplicate_point_id": pixel_duplicate.get("id"),
            "top_similarity_score": top_similarity_score,
            "pixel_hash": pixel_hash or None,
            "probable_duplicate": probable_duplicate,
            "tone_engine_used": True,
            "visual_intelligence_enabled": True,
        },
    }


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
        source="routers.vision.analyze_image_async",
        request_id=str(getattr(http_request.state, "request_id", "") or ""),
    )
    return {"success": True, "status": "queued", "task_id": task_id}
