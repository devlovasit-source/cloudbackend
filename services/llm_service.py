import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

from brain.tone.tone_engine import tone_engine

# =========================
# CONFIG
# =========================
load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

MODEL_FALLBACKS = [
    m.strip()
    for m in os.getenv(
        "OLLAMA_MODEL_FALLBACKS",
        "llama3.1:latest,llama3.1",
    ).split(",")
    if m.strip()
]

DEFAULT_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "1024"))
DEFAULT_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "300"))

# =========================
# SESSION (RETRY)
# =========================
session = requests.Session()
retries = Retry(total=2, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
session.mount("http://", HTTPAdapter(max_retries=retries))


# =========================
# CORE REQUEST
# =========================
def _call_ollama(payload, timeout=30):
    for model in [payload.get("model"), *MODEL_FALLBACKS]:
        try:
            payload["model"] = model
            payload["options"] = {
                "num_ctx": DEFAULT_NUM_CTX,
                "num_predict": DEFAULT_NUM_PREDICT,
                "temperature": 0.7,
            }

            res = session.post(f"{OLLAMA_URL}/generate", json=payload, timeout=timeout)

            if res.status_code == 200:
                return res.json()

        except Exception:
            continue

    return None


# =========================
# 🔥 WEATHER OVERLAY (RESTORED)
# =========================
def _select_weather_overlay(signals: dict) -> str:
    if not signals:
        return ""

    weather = str(signals.get("weather", "")).lower()

    overlays = {
        "summer": [
            "The lighter structure keeps this comfortable in heat.",
            "Breathable choices make this work well in warm weather."
        ],
        "rain": [
            "A slightly more structured base will handle weather shifts better.",
            "Avoid delicate fabrics here for practicality."
        ],
        "winter": [
            "Layering would add depth and structure here.",
            "A clean outer layer can elevate this instantly."
        ]
    }

    for key, options in overlays.items():
        if key in weather:
            import random
            return random.choice(options)

    return ""


# =========================
# 🔥 TEXT GENERATION
# =========================
def generate_text(prompt, user_profile=None, signals=None, model=None, timeout_seconds=30):

    tone = tone_engine.build_prompt_tone(user_profile, signals)

    full_prompt = f"""
You are AHVI, a premium AI fashion stylist.

Tone:
{tone.get("tone_instruction", "")}

Rules:
- Be natural and human
- Be concise but insightful
- Avoid generic advice
- Sound premium and confident

{prompt}
"""

    payload = {
        "model": model or DEFAULT_MODEL,
        "prompt": full_prompt,
        "stream": False,
    }

    data = _call_ollama(payload, timeout=timeout_seconds)

    if not data:
        return "none"

    response = data.get("response", "").strip()
    return tone_engine.apply(response, user_profile=user_profile, signals=signals)


# =========================
# 👔 OUTFIT EXPLANATION (UPGRADED)
# =========================
def generate_outfit_explanation(outfits, context="", user_profile=None, signals=None):

    overlay = _select_weather_overlay(signals or {})

    prompt = f"""
User wardrobe:
{context}

Outfit options:
{outfits}

Each option includes:
- label
- score
- aesthetic
- description

Explain:
- why each outfit works
- key differences
- when to wear each

Keep it premium and human.

Optional styling note:
{overlay}
"""

    return generate_text(prompt, user_profile, signals)


# =========================
# 🔥 ITEM LEVEL EXPLANATION (RESTORED)
# =========================
def generate_item_level_explanation(outfit, user_profile=None, signals=None):

    if not outfit:
        return []

    prompt = f"""
Outfit:
{outfit}

Explain EACH item:
- why it works
- role (base / highlight / balance)
- pairing logic

Return JSON:
[
  {{
    "item": "...",
    "reason": "...",
    "role": "...",
    "pairing": "..."
  }}
]
"""

    raw = generate_text(prompt, user_profile, signals)

    try:
        import json
        return json.loads(raw)
    except Exception:
        return []


# =========================
# 👗 STYLE ADVICE
# =========================
def generate_style_advice(user_input, wardrobe_summary, user_profile=None, signals=None):

    prompt = f"""
User request:
{user_input}

Wardrobe:
{wardrobe_summary}

Give practical styling advice.
Keep it sharp and useful.
"""

    return generate_text(prompt, user_profile, signals)


# =========================
# 🧠 SMART RESPONSE
# =========================
def generate_ai_response(user_input, outfits, wardrobe_items, user_profile=None, signals=None):

    wardrobe_summary = format_wardrobe_for_llm(wardrobe_items)

    if outfits:
        return generate_outfit_explanation(
            outfits,
            wardrobe_summary,
            user_profile,
            signals
        )

    return generate_style_advice(
        user_input,
        wardrobe_summary,
        user_profile,
        signals
    )


# =========================
# 👕 WARDROBE FORMATTER
# =========================
def format_wardrobe_for_llm(items):
    if not items:
        return "Wardrobe is empty."

    msg = "Wardrobe:\n"
    for item in items[:50]:
        msg += f"- {item.get('color')} {item.get('type')}\n"

    return msg
