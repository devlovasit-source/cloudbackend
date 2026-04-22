import base64
import asyncio
from typing import Any, Optional

import cv2
import numpy as np
import httpx


# =========================
# CACHE (LRU STYLE)
# =========================
_URL_HASH_CACHE: dict[str, str] = {}
_URL_HASH_CACHE_MAX = 2048


def _cache_get(url: str) -> str:
    return _URL_HASH_CACHE.get(url, "")


def _cache_set(url: str, value: str):
    if not url or not value:
        return

    _URL_HASH_CACHE[url] = value

    if len(_URL_HASH_CACHE) > _URL_HASH_CACHE_MAX:
        _URL_HASH_CACHE.pop(next(iter(_URL_HASH_CACHE)), None)


# =========================
# IMAGE DECODE
# =========================
def _decode_bytes(image_bytes: bytes):
    if not image_bytes:
        return None

    arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)


def _foreground_crop(img):
    if img is None:
        return None

    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.shape[2] == 4:
        alpha = img[:, :, 3]
        ys, xs = np.where(alpha > 10)

        if len(xs) and len(ys):
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            return img[y1:y2, x1:x2, :3]

        return img[:, :, :3]

    return img[:, :, :3]


# =========================
# CORE HASH FUNCTION (DHash)
# =========================
def compute_hash_from_bytes(image_bytes: bytes, size: int = 8) -> str:
    try:
        img = _decode_bytes(image_bytes)
        crop = _foreground_crop(img)

        if crop is None or crop.size == 0:
            return ""

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        resized = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]

        bits = diff.flatten()
        if bits.size == 0:
            return ""

        value = 0
        for b in bits:
            value = (value << 1) | int(b)

        # 🔥 FIX: pad hex properly (consistent length)
        width = (size * size + 3) // 4
        return f"{value:0{width}x}"

    except Exception:
        return ""


# =========================
# BASE64
# =========================
def compute_hash_from_base64(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""

    if "," in text:
        text = text.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(text, validate=True)
    except Exception:
        return ""

    return compute_hash_from_bytes(image_bytes)


# =========================
# URL (ASYNC)
# =========================
async def compute_hash_from_url(url: str, timeout: float = 6.0) -> str:
    if not url:
        return ""

    cached = _cache_get(url)
    if cached:
        return cached

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
            resp.raise_for_status()

            h = compute_hash_from_bytes(resp.content)

            if h:
                _cache_set(url, h)

            return h

    except Exception:
        return ""


# =========================
# 🔥 HAMMING DISTANCE (HEX SAFE)
# =========================
def hamming_distance_hex(h1: Any, h2: Any) -> Optional[int]:
    a = str(h1 or "").strip().lower()
    b = str(h2 or "").strip().lower()

    if not a or not b or len(a) != len(b):
        return None

    try:
        return (int(a, 16) ^ int(b, 16)).bit_count()
    except Exception:
        return None


# =========================
# BACKWARD COMPATIBILITY
# =========================
def hamming_distance(h1: str, h2: str) -> Optional[int]:
    return hamming_distance_hex(h1, h2)
# =========================
# 🔥 BACKWARD COMPATIBILITY (CRITICAL FIX)
# =========================

def compute_pixel_hash_from_base64(value: Any) -> str:
    return compute_hash_from_base64(value)
