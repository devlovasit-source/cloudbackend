import logging
import os
from typing import Dict, Any, List, Union

from brain.tone.tone_engine import tone_engine
from services.llm_service import (
    generate_outfit_explanation,
    generate_style_advice,
    generate_followup_suggestions,
)

logger = logging.getLogger("ahvi.response_assembler")


class ResponseAssembler:

    # =========================
    # MAIN ENTRY
    # =========================
    def assemble(self, merged_output: dict, context: dict = None) -> dict:
        context = context or {}
        response_type = merged_output.get("type")

        if response_type == "styling":
            return self._assemble_styling(merged_output, context)

        return self._assemble_multi_domain(merged_output, context)

    # =========================
    # 🔥 STYLING (ELITE)
    # =========================
    def _assemble_styling(self, merged_output: dict, context: dict) -> dict:

        data = merged_output.get("data", {})
        outfits = data.get("outfits", [])

        if not outfits:
            return self._wrap_response(
                "Nothing strong yet — want me to refine it?",
                context
            )

        user_profile = context.get("user_profile", {})
        style_dna = context.get("style_dna", {})
        signals = context.get("signals", {})

        best = outfits[0]
        items = best.get("items", [])

        # -------------------------
        # BASE VISUAL LOGIC
        # -------------------------
        base_text = self._build_visual_explanation(items, context)

        # -------------------------
        # 🔥 LLM ENHANCEMENT
        # -------------------------
        if self._llm_enabled():
            try:
                llm_text = generate_outfit_explanation(
                    outfits=outfits,
                    context=context,
                    user_profile=user_profile,
                    style_dna=style_dna,
                    signals=signals,
                )
                if llm_text:
                    base_text = llm_text
            except Exception as e:
                logger.warning("LLM styling failed: %s", e)

        # -------------------------
        # FINAL MESSAGE BUILD
        # -------------------------
        message = self._compose([
            self._proactive_prefix(signals),
            self._reaction(False),
            base_text
        ])

        message = tone_engine.apply(
            message,
            user_profile=user_profile,
            signals=signals,
        )

        # -------------------------
        # ACTIONS (CHIPS)
        # -------------------------
        actions = self._safe_actions(context, outfits)

        return self._wrap_response(message, context, actions=actions)

    # =========================
    # 🧠 MULTI DOMAIN
    # =========================
    def _assemble_multi_domain(self, merged_output: dict, context: dict) -> dict:

        data = merged_output.get("data", {})
        user_profile = context.get("user_profile", {})
        signals = context.get("signals", {})

        base_text = merged_output.get("message") or self._fallback(data)

        # optional polish
        if self._llm_enabled():
            try:
                improved = generate_style_advice(
                    user_input=base_text,
                    wardrobe_summary=str(context.get("wardrobe", "")),
                    user_profile=user_profile,
                    signals=signals,
                )
                if improved:
                    base_text = improved
            except Exception as e:
                logger.warning("LLM multi-domain failed: %s", e)

        message = self._compose([
            self._reaction(True),
            base_text,
            self._closer()
        ])

        message = tone_engine.apply(
            message,
            user_profile=user_profile,
            signals=signals,
        )

        return self._wrap_response(message, context)

    # =========================
    # 🔥 CORE BUILDERS
    # =========================
    def _wrap_response(self, text: str, context: dict, actions: List[str] = None) -> dict:
        return {
            "message": {
                "role": "assistant",
                "content": text.strip()
            },
            "chips": actions or [],
            "board_ids": context.get("board_ids"),
            "pack_id": context.get("pack_id"),
        }

    def _compose(self, parts: List[str]) -> str:
        parts = [p for p in parts if p]
        text = " ".join(parts)

        sentences = text.split(". ")
        if len(sentences) > 3:
            text = ". ".join(sentences[:3])

        return text.strip()

    # =========================
    # 🧠 VISUAL INTELLIGENCE
    # =========================
    def _build_visual_explanation(self, items, context):

        if not items:
            return "This is a clean look."

        colors = [i.get("color") for i in items if i]
        styles = [i.get("style") for i in items if i]

        parts = []

        if len(set(colors)) == 1:
            parts.append("The monochrome palette keeps it sharp.")
        else:
            parts.append("The colors balance well without clashing.")

        if len(set(styles)) == 1 and styles[0]:
            parts.append(f"It leans into a {styles[0]} aesthetic.")

        if context.get("occasion"):
            parts.append(f"It works naturally for {context.get('occasion')}.")

        return " ".join(parts)

    # =========================
    # 🧩 HELPERS
    # =========================
    def _safe_actions(self, context, outfits):
        try:
            return generate_followup_suggestions(context, outfits)
        except Exception:
            return ["Make it sharper", "More relaxed", "Change colors"]

    def _llm_enabled(self):
        return os.getenv("ENABLE_LLM_SYNTHESIS", "true").lower() in ("1", "true")

    def _proactive_prefix(self, signals):
        if signals.get("suggestion_type") == "morning_outfit":
            return "Here’s something for your morning."
        if signals.get("suggestion_type") == "evening_outfit":
            return "This fits your evening."
        return ""

    def _reaction(self, multi):
        return (
            "I’ve put this together for you."
            if multi else "This is a strong look."
        )

    def _closer(self):
        return "Want to refine this?"

    def _fallback(self, data: dict) -> str:
        if not data:
            return "I couldn’t find anything meaningful."

        chunks = []
        for k, v in data.items():
            if isinstance(v, dict):
                chunks.append(v.get("message") or v.get("summary") or "")
            else:
                chunks.append(str(v))

        return " ".join(chunks)


# singleton
response_assembler = ResponseAssembler()
