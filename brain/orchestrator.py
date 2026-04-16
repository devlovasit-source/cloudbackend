from typing import Any, Dict

from brain.nlu.intent_router import nlu_router
from brain.engines.style_graph_engine import style_graph_engine
from brain.engines.decision_engine import decision_engine
from brain.engines.palette_engine import PaletteEngine
from brain.engines.organize_engine import organize_engine
from brain.engines.meal_planner_engine import meal_planner_engine
from brain.engines.fitness_engine import fitness_engine

from brain.engines.outfit_engine import get_daily_outfits  # your existing
from brain.engines.style_rules_engine import style_engine

from brain.response.assembler import assemble_response
from brain.tone.tone_engine import apply_tone

palette_engine = PaletteEngine()


class Orchestrator:
    """
    🔥 SINGLE SOURCE OF TRUTH

    Flow:
    Intent → Context → Engine → Decision → Assembly → Tone
    """

    # =========================
    # ENTRY POINT
    # =========================
    def handle(self, user_input: str, user: Dict[str, Any]) -> Dict[str, Any]:

        # -------------------------
        # 1. INTENT
        # -------------------------
        intent_data = nlu_router.classify_intent(user_input)
        intent = intent_data.get("intent")
        slots = intent_data.get("slots", {})

        # -------------------------
        # 2. CONTEXT
        # -------------------------
        context = self._build_context(user, slots)

        # -------------------------
        # 3. ROUTE TO ENGINE
        # -------------------------
        if intent == "styling":
            result = self._handle_styling(context)

        elif intent == "meal_planning":
            result = meal_planner_engine.build_weekly_plan(context)

        elif intent == "health_wellness":
            result = fitness_engine.recommend_workout(context)

        elif intent == "finance_home":
            result = organize_engine.build_dashboard()

        else:
            result = self._fallback()

        # -------------------------
        # 4. RESPONSE ASSEMBLY
        # -------------------------
        response = assemble_response(result, context)

        # -------------------------
        # 5. TONE + PERSONALITY
        # -------------------------
        response["message"] = apply_tone(
            response.get("message", ""),
            context=context
        )

        return response

    # =========================
    # CONTEXT BUILDER
    # =========================
    def _build_context(self, user: Dict[str, Any], slots: Dict[str, Any]) -> Dict[str, Any]:

        return {
            "user_id": user.get("user_id"),
            "persona": user.get("persona", "default"),
            "style_dna": user.get("style_dna", {}),
            "wardrobe": user.get("wardrobe", []),
            "slots": slots,
            "occasion": slots.get("occasion"),
            "weather": slots.get("weather"),
        }

    # =========================
    # STYLING FLOW (CORE)
    # =========================
    def _handle_styling(self, context: Dict[str, Any]) -> Dict[str, Any]:

        wardrobe = context.get("wardrobe", [])

        if not wardrobe:
            return {"message": "Upload your wardrobe first."}

        # -------------------------
        # 1. STYLE GRAPH
        # -------------------------
        graph = style_graph_engine.build_graph({
            "tops": [i for i in wardrobe if i.get("category") == "top"],
            "bottoms": [i for i in wardrobe if i.get("category") == "bottom"],
            "shoes": [i for i in wardrobe if i.get("category") == "shoes"],
        })

        # -------------------------
        # 2. GENERATE OUTFITS
        # -------------------------
        outfit_data = get_daily_outfits({
            "user_id": context["user_id"],
            "wardrobe": wardrobe
        })

        outfits = outfit_data.get("outfits", [])

        # -------------------------
        # 3. APPLY STYLE RULES
        # -------------------------
        rules = style_engine.get_scoring_rules(
            context.get("style_dna"),
            context
        )

        enriched = []
        for outfit in outfits:
            score = outfit.get("score", 0)

            # simple rule boost example
            for item in outfit.get("items", []):
                if item.get("color") in rules["preferred_colors"]:
                    score += 1

            enriched.append({
                **outfit,
                "final_score": score
            })

        # -------------------------
        # 4. DECISION ENGINE
        # -------------------------
        selected, meta = decision_engine.rank_actions(
            candidates=enriched,
            context=context,
            top_n=3
        )

        # -------------------------
        # 5. PALETTE
        # -------------------------
        palette = palette_engine.select_palette({
            "event": context.get("occasion"),
            "microtheme": context.get("style_dna", {}).get("aesthetic")
        })

        return {
            "type": "styling",
            "outfits": selected,
            "palette": palette,
            "meta": meta
        }

    # =========================
    # FALLBACK
    # =========================
    def _fallback(self):
        return {
            "message": "Tell me what you need — outfit, meals, or plans?"
        }


# Singleton
orchestrator = Orchestrator()
