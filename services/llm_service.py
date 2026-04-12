import os
from dotenv import load_dotenv

from brain.tone.tone_engine import tone_engine

# =========================
# CONFIG
# =========================
load_dotenv()

# 🚫 OLLAMA DISABLED
OLLAMA_URL = None

DEFAULT_MODEL = "phi3"
MODEL_FALLBACKS = []
ALLOW_HEAVY_MODELS = False

DEFAULT_NUM_CTX = 1024
DEFAULT_NUM_PREDICT = 120


# =========================
# DISABLED REQUEST LAYER
# =========================
def safe_request(*args, **kwargs):
    return None


# =========================
# STYLIST GUIDANCE (UNCHANGED)
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
# TEXT GENERATION (DISABLED)
# =========================
def generate_text(
    prompt: str,
    options: dict = None,
    user_profile=None,
    signals=None,
    model: str | None = None,
    timeout_seconds: int | None = None,
) -> str:
    return "none"


# =========================
# CHAT COMPLETION (DISABLED)
# =========================
def chat_completion(
    messages: list,
    system_instruction: str = "",
    model: str = DEFAULT_MODEL,
    user_profile=None,
    signals=None,
    timeout_seconds: int | None = None,
) -> str:
    return "none"


# =========================
# WARDROBE FORMATTER (UNCHANGED)
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
# OUTFIT EXPLANATION (SAFE FALLBACK)
# =========================
def generate_outfit_explanation(outfits: list, context: str = "", user_profile=None, signals=None):
    return "Clean, balanced outfit with good color coordination and versatility."


# =========================
# STYLE ADVICE (SAFE FALLBACK)
# =========================
def generate_style_advice(user_input: str, wardrobe_summary: str, user_profile=None, signals=None):
    return "Try combining complementary colors and ensure proper fit for a polished look."


# =========================
# SMART RESPONSE GENERATOR (SAFE)
# =========================
def generate_ai_response(user_input: str, outfits: list, wardrobe_items: list, user_profile=None, signals=None):
    if outfits:
        return generate_outfit_explanation(outfits, "", user_profile=user_profile, signals=signals)

    return generate_style_advice(user_input, "", user_profile=user_profile, signals=signals)
