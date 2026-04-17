# =========================
# OUTFIT EXPLANATION (UPGRADED)
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


def generate_outfit_explanation(outfits: list, context: str = "", user_profile=None, signals=None):
    overlay = _select_weather_overlay(signals or {})

    prompt = f"""
User wardrobe:
{context}

Outfit options:
{outfits}

Each option includes:
- label (Easy Win / Sharp Upgrade / Statement Move)
- score
- aesthetic (vibe, color story, occasion)
- description

Explain:
- why each outfit works
- key difference between them
- when to wear each

Keep it concise, premium, and human.

Optional styling note:
{overlay}
"""
    return generate_text(
        prompt,
        user_profile=user_profile,
        signals=signals,
        usecase="styling"
    )


# =========================
# 🔥 ITEM LEVEL EXPLANATION (NEW)
# =========================
def generate_item_level_explanation(outfit: dict, user_profile=None, signals=None):
    if not outfit:
        return "No outfit selected."

    items = outfit.get("items", [])
    score = outfit.get("score")
    aesthetic = outfit.get("aesthetic", {})
    description = outfit.get("description", "")
    reasons = outfit.get("reasons", [])

    prompt = f"""
Outfit:
{items}

Score:
{score}

Aesthetic:
{aesthetic}

Description:
{description}

Overall reasons:
{reasons}

Explain EACH item:
- why it works
- what role it plays (base / highlight / balance)
- how it pairs with others

Return JSON only:
[
  {{
    "item": "white shirt",
    "reason": "keeps the outfit clean",
    "role": "base",
    "pairing": "balances darker bottoms"
  }}
]
"""
    return generate_text(
        prompt,
        user_profile=user_profile,
        signals=signals,
        usecase="styling"
    )


# =========================
# STYLE ADVICE (UNCHANGED)
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
    return generate_text(prompt, user_profile=user_profile, signals=signals, usecase="styling")


# =========================
# SMART RESPONSE GENERATOR (UPGRADED)
# =========================
def generate_ai_response(user_input: str, outfits: list, wardrobe_items: list, user_profile=None, signals=None):
    wardrobe_summary = format_wardrobe_for_llm(wardrobe_items)

    if outfits:
        return generate_outfit_explanation(
            outfits,
            wardrobe_summary,
            user_profile=user_profile,
            signals=signals
        )

    return generate_style_advice(
        user_input,
        wardrobe_summary,
        user_profile=user_profile,
        signals=signals
    )
