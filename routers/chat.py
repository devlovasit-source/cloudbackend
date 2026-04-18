from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any
import logging
import time
import concurrent.futures

from deep_translator import GoogleTranslator

from brain.orchestrator import ahvi_orchestrator
from services.appwrite_proxy import AppwriteProxy
from services.task_queue import enqueue_task
from services.weather_service import get_hourly_weather
from brain.response_validator import validate_orchestrator_response

try:
    from worker import run_heavy_audio_task
except Exception:
    run_heavy_audio_task = None

router = APIRouter()
logger = logging.getLogger("ahvi.routers.chat")

_CHAT_CACHE = {}

# =========================
# HELPERS
# =========================
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

def _is_greeting(text: str) -> bool:
    return text.lower().strip() in ["hi", "hello", "hey"]

def _is_fast_wardrobe_query(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in ["how many", "count", "total"]) and \
           any(k in t for k in ["wardrobe", "closet", "shirts", "pants", "shoes"])

def _fast_wardrobe_response(user_id: str):
    try:
        docs = AppwriteProxy().list_documents("outfits", user_id=user_id, limit=100)
    except Exception:
        docs = []

    return {
        "success": True,
        "message": f"You have {len(docs)} items in your wardrobe.",
        "type": "stats",
        "cards": [],
        "data": {"total_items": len(docs)},
        "meta": {"fast_path": True},
    }

# =========================
# MODELS
# =========================
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
    messages: List[Message]
    language: str = "en"
    current_memory: Any = {}
    user_profile: Dict[str, Any] = {}
    user_id: str | None = None
    userID: str | None = None


# =========================
# MAIN CHAT
# =========================
@router.post("/text")
def text_chat(request: TextChatRequest, http_request: Request):

    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages")

    user_input = request.messages[-1].content.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="Empty message")

    user_id = request.user_id or request.userID or "user_1"

    # -------------------------
    # CHIP SUPPORT (🔥 NEW)
    # -------------------------
    chip = http_request.query_params.get("chip")

    # -------------------------
    # CACHE
    # -------------------------
    cache_key = _cache_key(user_input + (chip or ""), user_id)
    cached = _get_cached(_CHAT_CACHE, cache_key)
    if cached:
        return cached

    # -------------------------
    # FAST PATHS
    # -------------------------
    if _is_greeting(user_input):
        return validate_orchestrator_response({
            "success": True,
            "message": "Hey! I can help you style outfits or plan your looks 👌",
            "cards": [],
            "meta": {"mode": "greeting"},
        })

    if _is_fast_wardrobe_query(user_input):
        return validate_orchestrator_response(
            _fast_wardrobe_response(user_id)
        )

    # -------------------------
    # TRANSLATION
    # -------------------------
    try:
        lang = (request.language or "en").lower()
        if lang in ["hi", "te"]:
            english_input = GoogleTranslator(source=lang, target="en").translate(user_input)
            target_lang = lang
        else:
            english_input = user_input
            target_lang = "en"
    except Exception:
        english_input = user_input
        target_lang = "en"

    # -------------------------
    # WEATHER
    # -------------------------
    weather = {}
    try:
        loc = request.user_profile.get("location", {})
        if loc.get("lat") and loc.get("lon"):
            weather = get_hourly_weather(loc["lat"], loc["lon"])
    except Exception:
        pass

    # -------------------------
    # LOAD WARDROBE (🔥 CRITICAL)
    # -------------------------
    try:
        wardrobe_docs = AppwriteProxy().list_documents(
            "outfits",
            user_id=user_id,
            limit=200
        )
    except Exception:
        wardrobe_docs = []

    # -------------------------
    # ORCHESTRATOR
    # -------------------------
    def run():
        return ahvi_orchestrator.handle(
            user_input=english_input,
            user={
                "user_id": user_id,
                "profile": request.user_profile,
                "memory": request.current_memory,
                "wardrobe": wardrobe_docs,
                "refinement": chip,  # 🔥 key integration
            }
        )

    try:
        with concurrent.futures.ThreadPoolExecutor() as ex:
            result = ex.submit(run).result(timeout=8)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # -------------------------
    # HANDLE RESPONSE
    # -------------------------
    assembled = result.get("message")

    if isinstance(assembled, dict):
        message_text = assembled.get("content") or assembled.get("message")
        actions = result.get("chips", []) or assembled.get("chips", [])
    else:
        message_text = assembled
        actions = result.get("actions", [])

    # -------------------------
    # TRANSLATE BACK
    # -------------------------
    try:
        if target_lang != "en" and message_text:
            message_text = GoogleTranslator(source="en", target=target_lang).translate(message_text)
    except Exception:
        pass

    # -------------------------
    # AUDIO
    # -------------------------
    try:
        audio_job_id = enqueue_task(run_heavy_audio_task, args=[message_text, target_lang]) \
            if run_heavy_audio_task else "offline"
    except Exception:
        audio_job_id = "offline"

    # -------------------------
    # FINAL RESPONSE
    # -------------------------
    response = {
        "success": True,
        "message": message_text,
        "chips": actions,  # 🔥 important
        "cards": result.get("data", {}).get("boards", []),
        "data": result.get("data", {}),
        "meta": {
            **(result.get("meta") or {}),
            "weather": weather,
            "refinement": chip,
        },
        "audio_job_id": audio_job_id,
    }

    _set_cache(_CHAT_CACHE, cache_key, response)

    return validate_orchestrator_response(response)
