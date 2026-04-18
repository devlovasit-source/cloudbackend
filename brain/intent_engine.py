
import json
from typing import Dict, Any

from services.ai_gateway import generate_text
from backend.brain.nlu.intent_router import nlu_router


# =========================
# PROMPT
# =========================
INTENT_PROMPT = """
You are an intent classification engine for an AI stylist and organizer app.

Return ONLY JSON.

Schema:
{
  "intent": "daily_dependency | daily_outfit | occasion_outfit | explore_styles | wardrobe_query | try_on | organize_hub | plan_pack | general",
  "slots": {
    "occasion": "string or null",
    "style": "string or null",
    "vibe": "string or null",
    "time": "morning | midday | afternoon | evening | night | null",
    "module": "life_boards | meal_planner | medicines | bills | calendar | workout | skincare | contacts | life_goals | null"
  },
  "confidence": 0.0-1.0
}

User:
"""


# =========================
# SAFE PARSE
# =========================
def _safe_parse(text: str) -> Dict[str, Any]:
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return {"intent": "general", "slots": {}, "confidence": 0.3}


# =========================
# 🔥 PREDICTIVE LAYER
# =========================
def _predict_intent(context: Dict[str, Any]) -> Dict[str, Any]:

    session = context.get("session", {})
    memory = context.get("user_memory", {})
    signals = context.get("signals", {})

    scores = {
        "daily_outfit": 0.0,
        "explore_styles": 0.0,
        "plan_pack": 0.0,
        "refinement": 0.0,
        "general": 0.0
    }

    # SESSION (strong)
    if session.get("refinement_history"):
        scores["refinement"] += 1.5

    if session.get("intent"):
        scores[session["intent"]] += 1.2

    # USER BEHAVIOR
    if signals.get("interaction_type") == "explore":
        scores["explore_styles"] += 1.0

    # MEMORY
    style_mem = memory.get("style_memory", {})
    if style_mem.get("occasions"):
        scores["plan_pack"] += 0.6

    # CONTEXT
    if context.get("occasion"):
        scores["plan_pack"] += 1.0

    best = max(scores, key=scores.get)

    return {
        "intent": best,
        "confidence": scores[best]
    }


# =========================
# 🔥 HYBRID RESOLUTION
# =========================
def _resolve_intent(
    router: Dict[str, Any],
    detected: Dict[str, Any],
    predicted: Dict[str, Any],
    context: Dict[str, Any]
) -> Dict[str, Any]:

    r_conf = float(router.get("confidence", 0.0))
    d_conf = float(detected.get("confidence", 0.0))
    p_conf = float(predicted.get("confidence", 0.0))

    r_intent = router.get("intent", "general")
    d_intent = detected.get("intent", "general")
    p_intent = predicted.get("intent", "general")

    final = d_intent
    source = "llm"

    # -------------------------
    # RULE 1: ROUTER STRONG
    # -------------------------
    if r_conf >= 0.9:
        final = r_intent
        source = "router_strong"

    # -------------------------
    # RULE 2: LLM STRONG
    # -------------------------
    elif d_conf >= 0.75:
        final = d_intent
        source = "llm_strong"

    # -------------------------
    # RULE 3: BEHAVIOR OVERRIDE
    # -------------------------
    elif p_conf >= 1.2:
        final = p_intent
        source = "predicted_override"

    # -------------------------
    # RULE 4: SOFT BLEND
    # -------------------------
    elif p_conf > d_conf:
        final = p_intent
        source = "predicted_soft"

    # -------------------------
    # RULE 5: SESSION PRIORITY
    # -------------------------
    if context.get("session", {}).get("refinement_history"):
        final = "refinement"
        source = "session_override"

    # -------------------------
    # SLOT MERGE
    # -------------------------
    slots = {
        **detected.get("slots", {}),
        **router.get("slots", {})
    }

    return {
        "intent": final,
        "slots": slots,
        "confidence": max(r_conf, d_conf, p_conf),
        "source": source,
        "router": router,
        "detected": detected,
        "predicted": predicted
    }


# =========================
# MAIN API
# =========================
def detect_intent(
    user_text: str,
    history=None,
    context: Dict[str, Any] = None,
    model: str | None = None
) -> Dict[str, Any]:

    if not user_text:
        return {"intent": "general", "slots": {}, "confidence": 0.0}

    context = context or {}

    # -------------------------
    # 1. ROUTER (FAST)
    # -------------------------
    router_result = nlu_router.classify_intent(user_text)

    # -------------------------
    # 2. LLM DETECTION
    # -------------------------
    detected = None

    if router_result.get("confidence", 0.0) < 0.85:
        prompt = INTENT_PROMPT + user_text

        response = generate_text(
            prompt,
            options={"temperature": 0.2, "num_predict": 200},
            usecase="intent",
            model=model,
        )

        detected = _safe_parse(response)
    else:
        detected = {
            "intent": router_result["intent"],
            "slots": router_result["slots"],
            "confidence": router_result["confidence"]
        }

    # -------------------------
    # HISTORY CONTEXT
    # -------------------------
    if history:
        last = history[-1]
        if detected["intent"] == "general" and last.get("intent"):
            detected["intent"] = last.get("intent")
            detected["confidence"] = max(detected.get("confidence", 0.0), 0.6)

    # -------------------------
    # 3. PREDICT
    # -------------------------
    predicted = _predict_intent(context)

    # -------------------------
    # 4. RESOLVE
    # -------------------------
    resolved = _resolve_intent(
        router=router_result,
        detected=detected,
        predicted=predicted,
        context=context
    )

    return resolved
