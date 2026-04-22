import hashlib
import json
import os
from typing import Any, Dict, List


_CACHE: Dict[str, Dict[str, Any]] = {}


def _brain_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _get_bank(path: str) -> Dict[str, Any]:
    if path in _CACHE:
        return _CACHE[path]
    data = _load_json(path)
    _CACHE[path] = data
    return data


def _stable_index(key: str, n: int) -> int:
    if n <= 0:
        return 0
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % n


def pick_bank_phrase(*, bank_path: str, category: str, key: str) -> str:
    bank = _get_bank(bank_path)
    cats = (
        (((bank.get("styling_intelligence") or {}) or {}).get(os.path.basename(bank_path).replace(".json", "")) or {})
        if isinstance(bank.get("styling_intelligence"), dict)
        else {}
    )
    # Some banks have a nested key (e.g. color_harmony_bank_v1); fall back to first dict value.
    if not cats and isinstance(bank.get("styling_intelligence"), dict):
        root = bank["styling_intelligence"]
        if isinstance(root, dict):
            for v in root.values():
                if isinstance(v, dict):
                    cats = v
                    break

    categories = cats.get("categories") if isinstance(cats, dict) else {}
    phrases = categories.get(category) if isinstance(categories, dict) else None
    if not isinstance(phrases, list) or not phrases:
        return ""
    idx = _stable_index(f"{key}:{category}", len(phrases))
    text = str(phrases[idx] or "").strip()
    return text


def color_harmony_snippet(score_hint: float, *, key: str) -> str:
    bank_path = os.path.join(_brain_dir(), "banks", "foundational", "color_harmony_bank.json")
    category = "positive_harmony" if score_hint >= 0.6 else "constructive_flat_or_clashing"
    return pick_bank_phrase(bank_path=bank_path, category=category, key=key)


def weather_overlay_snippet(weather_mode: str, *, key: str) -> str:
    bank_path = os.path.join(_brain_dir(), "banks", "contextual", "season_weather_overlays_bank_v1.json")
    weather_mode = str(weather_mode or "").strip().lower()
    if weather_mode in ("hot", "summer", "heat", "warm"):
        category = "summer_heat"
    elif weather_mode in ("cold", "winter", "chilly"):
        category = "winter_layering"
    else:
        category = "mild_weather"
    return pick_bank_phrase(bank_path=bank_path, category=category, key=key)
