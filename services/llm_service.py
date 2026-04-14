import os
import time
from threading import BoundedSemaphore, Lock

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from brain.tone.tone_engine import tone_engine

# =========================
# CONFIG
# =========================
load_dotenv()

OLLAMA_URL = str(os.getenv("OLLAMA_URL", "http://localhost:11434/api") or "").strip().rstrip("/")
if not OLLAMA_URL.endswith("/api"):
    OLLAMA_URL = f"{OLLAMA_URL}/api"

DEFAULT_MODEL = str(os.getenv("OLLAMA_TEXT_MODEL", os.getenv("OLLAMA_MODEL", "phi3:latest")) or "").strip()
MODEL_FALLBACKS = [
    m.strip()
    for m in str(os.getenv("OLLAMA_MODEL_FALLBACKS", "phi3:latest,phi3,tinyllama") or "").split(",")
    if m.strip()
]
ALLOW_HEAVY_MODELS = str(os.getenv("ALLOW_HEAVY_MODELS", "false")).strip().lower() in {"1", "true", "yes", "on"}
PIN_TEXT_MODEL = str(os.getenv("OLLAMA_PIN_TEXT_MODEL", "true")).strip().lower() in {"1", "true", "yes", "on"}

DEFAULT_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
DEFAULT_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "160"))
DEFAULT_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
DEFAULT_TOP_P = float(os.getenv("OLLAMA_TOP_P", "0.85"))
DEFAULT_REPEAT_PENALTY = float(os.getenv("OLLAMA_REPEAT_PENALTY", "1.15"))
MAX_PROMPT_CHARS = int(os.getenv("OLLAMA_MAX_PROMPT_CHARS", "7000"))
MAX_INFLIGHT_REQUESTS = max(1, int(os.getenv("OLLAMA_MAX_INFLIGHT_REQUESTS", "2")))
QUEUE_WAIT_SECONDS = max(0.1, float(os.getenv("OLLAMA_QUEUE_WAIT_SECONDS", "6")))

_request_slots = BoundedSemaphore(MAX_INFLIGHT_REQUESTS)
_metrics_lock = Lock()
_metrics = {
    "total_requests": 0,
    "failed_requests": 0,
    "queued_requests": 0,
    "queue_rejections": 0,
    "in_flight": 0,
    "peak_in_flight": 0,
    "last_latency_ms": 0,
    "last_error": "",
    "last_model": "",
}


# =========================
# SESSION WITH RETRIES
# =========================
session = requests.Session()
retries = Retry(
    total=2,
    backoff_factor=0.4,
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["POST"],
)
session.mount("http://", HTTPAdapter(max_retries=retries))
session.mount("https://", HTTPAdapter(max_retries=retries))


def _is_heavy_model(model_name: str) -> bool:
    name = str(model_name or "").lower()
    heavy_tokens = [":7b", ":8b", ":9b", ":13b", ":14b", ":32b", ":70b", "qwen2.5-vl", "llava"]
    return any(token in name for token in heavy_tokens)


def _model_candidates(requested_model: str | None) -> list[str]:
    if PIN_TEXT_MODEL:
        model = str(requested_model or DEFAULT_MODEL).strip()
        if not model:
            return []
        if not ALLOW_HEAVY_MODELS and _is_heavy_model(model):
            return []
        return [model]

    ordered: list[str] = []
    for candidate in [requested_model, DEFAULT_MODEL, *MODEL_FALLBACKS]:
        model = str(candidate or "").strip()
        if not model:
            continue
        if not ALLOW_HEAVY_MODELS and _is_heavy_model(model):
            continue
        if model not in ordered:
            ordered.append(model)
    return ordered


def _merged_options(incoming: dict | None) -> dict:
    merged = {
        "num_ctx": DEFAULT_NUM_CTX,
        "num_predict": DEFAULT_NUM_PREDICT,
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "repeat_penalty": DEFAULT_REPEAT_PENALTY,
    }
    if incoming:
        merged.update(incoming)
    return merged


def _usecase_options(usecase: str | None, incoming: dict | None) -> dict:
    case = str(usecase or "general").strip().lower()
    base = _merged_options(incoming)

    if case == "intent":
        # Keep intent classification deterministic.
        base.setdefault("num_predict", 180)
        base["temperature"] = min(float(base.get("temperature", DEFAULT_TEMPERATURE)), 0.12)
        base["top_p"] = min(float(base.get("top_p", DEFAULT_TOP_P)), 0.75)
        base["repeat_penalty"] = max(float(base.get("repeat_penalty", DEFAULT_REPEAT_PENALTY)), 1.18)
        return base

    if case in {"general", "styling"}:
        base["temperature"] = min(float(base.get("temperature", DEFAULT_TEMPERATURE)), 0.25)
        base["top_p"] = min(float(base.get("top_p", DEFAULT_TOP_P)), 0.9)
        base["repeat_penalty"] = max(float(base.get("repeat_penalty", DEFAULT_REPEAT_PENALTY)), 1.12)
        return base

    return base


def _grounding_rules() -> str:
    return """
Grounding and anti-hallucination rules:
- Do not invent wardrobe items, counts, weather, locations, prices, or user facts.
- If required data is missing, say it is unavailable and ask one concise follow-up question.
- Prefer short factual answers over creative elaboration.
- Never present guesses as facts.
"""


def _trim_prompt(text: str) -> str:
    raw = str(text or "")
    if len(raw) <= MAX_PROMPT_CHARS:
        return raw
    return raw[-MAX_PROMPT_CHARS:]


def _metrics_begin() -> float:
    now = time.perf_counter()
    with _metrics_lock:
        _metrics["total_requests"] += 1
    return now


def _metrics_mark_wait() -> None:
    with _metrics_lock:
        _metrics["queued_requests"] += 1


def _metrics_mark_rejected() -> None:
    with _metrics_lock:
        _metrics["queue_rejections"] += 1
        _metrics["failed_requests"] += 1
        _metrics["last_error"] = "queue_timeout"


def _metrics_enter() -> None:
    with _metrics_lock:
        _metrics["in_flight"] += 1
        if _metrics["in_flight"] > _metrics["peak_in_flight"]:
            _metrics["peak_in_flight"] = _metrics["in_flight"]


def _metrics_exit(latency_ms: int) -> None:
    with _metrics_lock:
        _metrics["in_flight"] = max(0, int(_metrics.get("in_flight", 0)) - 1)
        _metrics["last_latency_ms"] = max(0, int(latency_ms))


def _metrics_fail(message: str) -> None:
    with _metrics_lock:
        _metrics["failed_requests"] += 1
        _metrics["last_error"] = str(message or "")[:500]


def _metrics_model(model: str) -> None:
    with _metrics_lock:
        _metrics["last_model"] = str(model or "")


def get_runtime_metrics() -> dict:
    with _metrics_lock:
        snapshot = dict(_metrics)
    snapshot.update(
        {
            "config": {
                "default_model": DEFAULT_MODEL,
                "pin_text_model": PIN_TEXT_MODEL,
                "num_ctx": DEFAULT_NUM_CTX,
                "temperature": DEFAULT_TEMPERATURE,
                "max_inflight": MAX_INFLIGHT_REQUESTS,
                "queue_wait_seconds": QUEUE_WAIT_SECONDS,
            }
        }
    )
    return snapshot


# =========================
# REQUEST LAYER
# =========================
def safe_request(endpoint: str, payload: dict, timeout: int = 30):
    started = _metrics_begin()
    candidates = _model_candidates(payload.get("model"))
    if not candidates:
        _metrics_fail("no_model_candidates")
        return None

    if not _request_slots.acquire(blocking=False):
        _metrics_mark_wait()
        if not _request_slots.acquire(timeout=QUEUE_WAIT_SECONDS):
            _metrics_mark_rejected()
            return None
    _metrics_enter()

    last_error = ""

    try:
        for model in candidates:
            local_payload = dict(payload)
            local_payload["model"] = model
            local_payload["options"] = _merged_options(local_payload.get("options"))
            try:
                response = session.post(
                    f"{OLLAMA_URL}/{endpoint}",
                    json=local_payload,
                    timeout=timeout,
                )
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, dict):
                        data["_model_used"] = model
                    _metrics_model(model)
                    return data
                last_error = f"{response.status_code}: {response.text}"
                if response.status_code == 404:
                    continue
            except Exception as exc:
                last_error = str(exc)
                continue

        if last_error:
            _metrics_fail(last_error)
            print(f"[llm_service] Ollama request failed ({endpoint}): {last_error}")
        return None
    finally:
        _metrics_exit(latency_ms=int((time.perf_counter() - started) * 1000))
        _request_slots.release()


# =========================
# STYLIST GUIDANCE
# =========================
def _stylist_guidance(user_profile=None, signals=None) -> str:
    user_profile = user_profile or {}
    signals = signals or {}
    context_mode = str(signals.get("context_mode", "general")).lower()
    if context_mode != "styling":
        return ""

    preferred_colors = user_profile.get("preferred_colors", user_profile.get("colors", []))
    style = user_profile.get("style", "")
    body_type = user_profile.get("body_type", "")
    budget = user_profile.get("budget", "")

    return f"""
Advanced Stylist Rules:
- Prioritize occasion, weather, and comfort first.
- Use wardrobe-aware recommendations and avoid generic trends.
- Give one best choice first, then one alternative.
- Add one practical upgrade (accessory, layer, or color swap).
- Mention confidence rationale in plain language.
- Keep output actionable and premium.

User style profile:
- style: {style}
- preferred colors: {preferred_colors}
- body type: {body_type}
- budget: {budget}
"""


# =========================
# TEXT GENERATION
# =========================
def generate_text(
    prompt: str,
    options: dict = None,
    user_profile=None,
    signals=None,
    model: str | None = None,
    timeout_seconds: int | None = None,
    usecase: str | None = None,
) -> str:
    if not prompt:
        return "none"

    tone = tone_engine.build_prompt_tone(user_profile, signals)
    full_prompt = f"""
You are AHVI, a premium AI fashion stylist.

Tone Instructions:
{tone.get("tone_instruction", "")}

Guidelines:
- Be natural and human
- Keep responses concise
- Sound confident but not arrogant
- Avoid robotic phrasing

{_stylist_guidance(user_profile=user_profile, signals=signals)}
{_grounding_rules()}

{prompt}
"""
    payload = {
        "model": model or DEFAULT_MODEL,
        "prompt": _trim_prompt(full_prompt),
        "stream": False,
    }
    payload["options"] = _usecase_options(usecase=usecase, incoming=options)

    data = safe_request("generate", payload, timeout=int(timeout_seconds or 30))
    if not data:
        return "none"

    response = str(data.get("response", "")).strip() or "none"
    try:
        response = tone_engine.apply(response, user_profile=user_profile, signals=signals)
    except Exception:
        pass
    return response


# =========================
# CHAT COMPLETION
# =========================
def chat_completion(
    messages: list,
    system_instruction: str = "",
    model: str = DEFAULT_MODEL,
    user_profile=None,
    signals=None,
    timeout_seconds: int | None = None,
    usecase: str | None = None,
) -> str:
    if not messages:
        return "I did not catch that."

    tone = tone_engine.build_prompt_tone(user_profile, signals)
    system_msg = f"""
You are AHVI, an AI fashion stylist.

Tone:
{tone.get("tone_instruction", "")}

Rules:
- Speak naturally
- Keep it concise
- Be stylish and practical

{_stylist_guidance(user_profile=user_profile, signals=signals)}
{_grounding_rules()}
"""
    if system_instruction:
        system_msg += "\n" + str(system_instruction)[:2000]

    combined_prompt = system_msg + "\n\n"
    for msg in (messages or [])[-10:]:
        role = str(msg.get("role", "user")).upper()
        content = str(msg.get("content", ""))[:4000]
        if content:
            combined_prompt += f"{role}: {content}\n"
    combined_prompt += "ASSISTANT:"

    payload = {
        "model": model or DEFAULT_MODEL,
        "prompt": combined_prompt,
        "stream": False,
    }
    payload["options"] = _usecase_options(usecase=usecase, incoming=None)
    data = safe_request("generate", payload, timeout=int(timeout_seconds or 45))
    if not data:
        return "I am having trouble thinking right now."

    response = str(data.get("response", "")).strip()
    try:
        response = tone_engine.apply(response, user_profile=user_profile, signals=signals)
    except Exception:
        pass
    return response or "Something went wrong."


# =========================
# WARDROBE FORMATTER
# =========================
def format_wardrobe_for_llm(items):
    if not items:
        return "The user's wardrobe is empty."

    msg = "User wardrobe:\n"

    for item in items[:50]:
        category = item.get("category_group", "")
        sub = item.get("subcategory", "")
        color = item.get("colors", {}).get("primary", "") if isinstance(item.get("colors"), dict) else item.get("color", "")
        msg += f"- {color} {sub} ({category})\n"

    return msg


# =========================
# OUTFIT EXPLANATION
# =========================
def generate_outfit_explanation(outfits: list, context: str = "", user_profile=None, signals=None):
    prompt = f"""
User wardrobe:
{context}

Outfits:
{outfits}

Explain:
- why these outfits work
- when to wear them

Keep it short (2-3 lines).
"""
    return generate_text(prompt, user_profile=user_profile, signals=signals)


# =========================
# STYLE ADVICE
# =========================
def generate_style_advice(user_input: str, wardrobe_summary: str, user_profile=None, signals=None):
    prompt = f"""
User request:
{user_input}

Wardrobe:
{wardrobe_summary}

Give practical styling advice using available wardrobe.
Keep it concise and helpful.
"""
    return generate_text(prompt, user_profile=user_profile, signals=signals)


# =========================
# SMART RESPONSE GENERATOR
# =========================
def generate_ai_response(user_input: str, outfits: list, wardrobe_items: list, user_profile=None, signals=None):
    wardrobe_summary = format_wardrobe_for_llm(wardrobe_items)

    if outfits:
        return generate_outfit_explanation(outfits, wardrobe_summary, user_profile=user_profile, signals=signals)

    return generate_style_advice(user_input, wardrobe_summary, user_profile=user_profile, signals=signals)
