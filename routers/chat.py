from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any
import re
import os
import logging
import time
import concurrent.futures

from deep_translator import GoogleTranslator

try:
    from worker import run_heavy_audio_task
except Exception:
    run_heavy_audio_task = None

from brain.orchestrator import ahvi_orchestrator
from brain.outfit_pipeline import save_feedback
from services.appwrite_proxy import AppwriteProxy
try:
    from services.job_tracker import job_tracker
except Exception:
    job_tracker = None
from services.task_queue import enqueue_task

# 🔥 NEW
from services.weather_service import get_hourly_weather

router = APIRouter()
logger = logging.getLogger("ahvi.routers.chat")

_CHAT_CACHE = {}
_WEATHER_CACHE = {}


def lightweight_chat(text: str) -> str:
    prompt = str(text or "").strip()
    if not prompt:
        return "Hey, what is on your mind today?"
    return "I can help with style, planning, and wardrobe advice. Tell me what you want to solve."

def _cache_key(text, user_id):
    return f"{user_id}:{text.lower().strip()}"

def _get_cached(cache, key, ttl=60):
    item = cache.get(key)
    if not item:
        return None
    if time.time() - item["time"] > ttl:
        del cache[key]
        return None
    return item["value"]

def _set_cache(cache, key, value):
    cache[key] = {"value": value, "time": time.time()}

def _build_history(messages: List["Message"]) -> List[Dict[str, Any]]:
    history: List[Dict[str, Any]] = []
    for msg in messages[-8:]:
        role = str(getattr(msg, "role", "user")).lower()
        content = str(getattr(msg, "content", "")).strip()
        if not content:
            continue
        history.append({"role": role, "text": content[:500]})
    return history


def _normalize_memory_history(events: Any, max_items: int = 12) -> List[Dict[str, Any]]:
    if not isinstance(events, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for event in events[-max_items:]:
        if not isinstance(event, dict):
            continue
        row: Dict[str, Any] = {}
        if event.get("intent"):
            row["intent"] = str(event.get("intent"))[:80]
        if isinstance(event.get("slots"), dict):
            row["slots"] = event.get("slots")
        if event.get("role"):
            row["role"] = str(event.get("role"))[:32]
        if event.get("text"):
            row["text"] = str(event.get("text"))[:500]
        if row:
            normalized.append(row)
    return normalized


def _is_fast_wardrobe_count_query(text: str) -> bool:
    lowered = str(text or "").lower()
    count_words = ["how many", "count", "number of", "total", "do i have"]
    wardrobe_words = [
        "wardrobe", "closet", "outfit", "outfits", "tops", "top", "shirts", "shirt",
        "pants", "trousers", "jeans", "bottoms", "shoes", "footwear", "dress",
        "dresses", "accessories", "jewelry", "bags", "bag",
    ]
    return any(k in lowered for k in count_words) and any(k in lowered for k in wardrobe_words)


def _fast_wardrobe_count_response(user_id: str, query_text: str) -> Dict[str, Any]:
    try:
        docs = AppwriteProxy().list_documents("outfits", user_id=user_id, limit=100)
    except Exception:
        docs = []

    counts = {"tops": 0, "bottoms": 0, "shoes": 0, "dresses": 0, "accessories": 0}
    for d in docs:
        category = str(d.get("category") or d.get("category_group") or "").lower()
        sub = str(d.get("sub_category") or d.get("subcategory") or "").lower()
        blob = f"{category} {sub}"
        if any(k in blob for k in ["top", "shirt", "blouse", "jacket", "blazer", "tee"]):
            counts["tops"] += 1
        elif any(k in blob for k in ["bottom", "pant", "trouser", "jean", "short", "skirt"]):
            counts["bottoms"] += 1
        elif any(k in blob for k in ["shoe", "sneaker", "heel", "boot", "sandal", "footwear"]):
            counts["shoes"] += 1
        elif "dress" in blob:
            counts["dresses"] += 1
        elif any(k in blob for k in ["accessory", "watch", "bag", "jewel", "necklace", "earring"]):
            counts["accessories"] += 1

    lowered = str(query_text or "").lower()
    if any(k in lowered for k in ["top", "tops", "shirt", "shirts", "blouse", "blouses"]):
        message = f"You have {counts['tops']} tops in your wardrobe."
    elif any(k in lowered for k in ["bottom", "bottoms", "pant", "pants", "trouser", "trousers", "jean", "jeans"]):
        message = f"You have {counts['bottoms']} bottoms in your wardrobe."
    elif any(k in lowered for k in ["shoe", "shoes", "footwear", "sneaker", "sneakers"]):
        message = f"You have {counts['shoes']} shoes in your wardrobe."
    else:
        total = len(docs)
        message = (
            f"You currently have {total} items: {counts['tops']} tops, {counts['bottoms']} bottoms, "
            f"{counts['shoes']} shoes, {counts['dresses']} dresses, and {counts['accessories']} accessories."
        )

    return {
        "success": True,
        "message": message,
        "board": "wardrobe",
        "type": "stats",
        "cards": [
            {"id": "tops", "title": "Tops", "kind": "stat", "value": counts["tops"]},
            {"id": "bottoms", "title": "Bottoms", "kind": "stat", "value": counts["bottoms"]},
            {"id": "shoes", "title": "Shoes", "kind": "stat", "value": counts["shoes"]},
            {"id": "dresses", "title": "Dresses", "kind": "stat", "value": counts["dresses"]},
            {"id": "accessories", "title": "Accessories", "kind": "stat", "value": counts["accessories"]},
        ],
        "data": {"counts": counts, "total_items": len(docs)},
        "meta": {"intent": "wardrobe_query", "domain": "wardrobe", "fast_path": True},
        "audio_job_id": "offline",
    }

def _detect_mode(text: str) -> str:
    t = text.lower().strip()

    if any(k in t for k in ["wear","outfit","dress","style","clothes","wardrobe","look"]):
        return "fashion"

    if t in ["hi","hello","hey"]:
        return "greeting"

    if any(k in t for k in ["how are","what is","who is","tell me","why","joke","explain"]):
        return "casual"

    return "fashion"

# -------------------------
# MODELS
# -------------------------
class Message(BaseModel):
    role: str = Field(..., min_length=1, max_length=24)
    content: str = Field(..., min_length=1, max_length=4000)

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        role = str(value or "").strip().lower()
        if role not in {"user", "assistant", "system"}:
            raise ValueError("role must be one of user/assistant/system")
        return role


class TextChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1, max_length=30)
    language: str = Field(default="en", min_length=2, max_length=8)
    current_memory: Any = Field(default_factory=dict)
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    user_id: str | None = None
    userID: str | None = None
    module_context: str | None = None


class OutfitFeedbackRequest(BaseModel):
    user_id: str
    feedback: str
    outfit: Dict[str, Any]


class OrganizeHubRequest(BaseModel):
    user_id: str
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    current_memory: Any = Field(default_factory=dict)
    include_counts: bool = False


class PlanPackRequest(BaseModel):
    user_id: str
    prompt: str
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    current_memory: Any = Field(default_factory=dict)


class DailyCardsRequest(BaseModel):
    user_id: str
    time_slot: str | None = None
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    current_memory: Any = Field(default_factory=dict)

@router.post("/text")
def text_chat(request: TextChatRequest, http_request: Request):

    # -------------------------
    # INPUT VALIDATION
    # -------------------------
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    user_input = (request.messages[-1].content or "").strip()

    if not user_input:
        raise HTTPException(status_code=400, detail="Empty message")

    user_id = request.user_id or request.userID or "user_1"

    # -------------------------
    # FAST PATH
    # -------------------------
    if _is_fast_wardrobe_count_query(user_input):
        return _fast_wardrobe_count_response(user_id, user_input)

    # -------------------------
    # CACHE
    # -------------------------
    cache_key = _cache_key(user_input, user_id)
    cached = _get_cached(_CHAT_CACHE, cache_key)
    if cached:
        return cached

    # -------------------------
    # LANGUAGE
    # -------------------------
    try:
        preferred_lang = (request.language or "en").lower()

        if preferred_lang in ("te", "hi"):
            english_input = GoogleTranslator(source=preferred_lang, target="en").translate(user_input)
            target_lang = preferred_lang
        else:
            english_input = user_input
            target_lang = "en"

    except Exception:
        english_input = user_input
        target_lang = "en"

    # -------------------------
    # HYBRID ROUTING
    # -------------------------
    mode = _detect_mode(english_input)

    if mode == "greeting":
        return {
            "success": True,
            "message": "Hey! I can help you style outfits or just chat 😊",
            "cards": [],
            "meta": {"mode": "greeting"},
            "audio_job_id": "offline",
        }

    if mode == "casual" and not request.module_context:
        try:
            return {
                "success": True,
                "message": lightweight_chat(english_input),
                "cards": [],
                "meta": {"mode": "casual"},
                "audio_job_id": "offline",
            }
        except Exception:
            pass

    # -------------------------
    # WEATHER
    # -------------------------
    weather_data = {}
    try:
        location = request.user_profile.get("location") or {}
        lat, lon = location.get("lat"), location.get("lon")

        if lat and lon:
            weather_data = get_hourly_weather(lat=lat, lon=lon)

    except Exception as e:
        logger.warning("weather lookup failed %s", e)

    # -------------------------
    # ORCHESTRATOR (TIMEOUT SAFE)
    # -------------------------
    history = _build_history(request.messages[:-1]) if len(request.messages) > 1 else []
    memory_history = request.current_memory.get("history", []) if isinstance(request.current_memory, dict) else []
    merged_history = _normalize_memory_history(memory_history) + history

    def run():
        return ahvi_orchestrator.run(
            text=english_input,
            user_id=user_id,
            context={
                "memory": request.current_memory,
                "user_profile": request.user_profile,
                "module_context": request.module_context,
                "history": merged_history[-20:],
                "weather": weather_data.get("condition"),
                "time_of_day": weather_data.get("time_of_day"),
            },
        )

    try:
        with concurrent.futures.ThreadPoolExecutor() as ex:
            result = ex.submit(run).result(timeout=8)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Orchestrator failed: {exc}")

    message = result.get("message") or ""

    # -------------------------
    # TRANSLATE BACK
    # -------------------------
    try:
        if target_lang != "en" and message:
            message = GoogleTranslator(source="en", target=target_lang).translate(message)
    except Exception:
        pass

    # -------------------------
    # AUDIO
    # -------------------------
    try:
        audio_job_id = (
            enqueue_task(run_heavy_audio_task, args=[message, target_lang])
            if run_heavy_audio_task else "offline"
        )
    except Exception:
        audio_job_id = "offline"

    # -------------------------
    # FINAL RESPONSE
    # -------------------------
    cards_payload = result.get("cards") or []
    if not isinstance(cards_payload, list):
        cards_payload = []

    board_ids_text = str(result.get("board_ids") or "")

    logger.info(
        "chat.text_response user_id=%s cards=%d",
        user_id,
        len(cards_payload),
    )

    response = {
        "success": True,
        "message": message,
        "board": result.get("board"),
        "type": result.get("type"),
        "cards": cards_payload,
        "board_ids": board_ids_text,
        "data": result.get("data") or {},
        "meta": {
            **(result.get("meta") or {}),
            "weather": weather_data,
            "history_used": len(merged_history[-20:])
        },
        "audio_job_id": audio_job_id,
    }

    # -------------------------
    # CACHE SAVE
    # -------------------------
    _set_cache(_CHAT_CACHE, cache_key, response)

    return response    
