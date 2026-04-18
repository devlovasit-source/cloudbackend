
import logging
import os
from typing import Dict, Any, List

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
        boards = data.get("boards", [])

        if not outfits:
            return self._wrap_response(
                "I need a bit more to work with — want me to refine it?",
                context,
                chips=["Show outfit ideas", "Try different vibe"]
            )

        user_profile = context.get("user_profile", {})
        signals = context.get("signals", {})

        # -------------------------
        # BASE VISUAL TEXT
        # -------------------------
        base_text = self._build_visual_explanation(outfits[0], context)

        # -------------------------
        # LLM ENHANCEMENT
        # -------------------------
        if self._llm_enabled():
            try:
                llm_text = generate_outfit_explanation(
                    outfits=outfits,
                    context=context,
                    user_profile=user_profile,
                    signals=signals,
                )

                if isinstance(llm_text, dict):
                    llm_text = llm_text.get("content", "")

                if llm_text:
                    base_text = llm_text

            except Exception as e:
                logger.warning("LLM styling failed: %s", e)

        # -------------------------
        # FINAL MESSAGE (PERSONALITY RESTORED)
        # -------------------------
        message = self._compose([
            self._proactive_prefix(signals),
            self._reaction(),
            base_text,
            self._closer()
        ])

        message = tone_engine.apply(
            message,
            user_profile=user_profile,
            signals=signals,
        )

        # -------------------------
        # CHIPS
        # -------------------------
        chips = self._safe_actions(context)

        return self._wrap_response(
            message,
            context,
            chips=chips,
            cards=boards,
            data={"outfits": outfits}
        )

    # =========================
    # 🧠 MULTI DOMAIN
    # =========================
    def _assemble_multi_domain(self, merged_output: dict, context: dict) -> dict:

        data = merged_output.get("data", {})
        user_profile = context.get("user_profile", {})
        signals = context.get("signals", {})

        base_text = merged_output.get("message") or self._fallback(data)

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
            self._reaction(),
            base_text,
            self._closer()
        ])

        message = tone_engine.apply(
            message,
            user_profile=user_profile,
            signals=signals,
        )

        return self._wrap_response(
            message,
            context,
            chips=self._safe_actions(context),
            data=data
        )

    # =========================
    # 🔥 RESPONSE WRAPPER (RESTORED CLEANLY)
    # =========================
    def _wrap_response(
        self,
        text: str,
        context: dict,
        chips: List[str] = None,
        cards: List[dict] = None,
        data: Dict[str, Any] = None
    ) -> dict:

        return {
            "message": {
                "role": "assistant",
                "content": text.strip()
            },
            "chips": chips or [],
            "cards": cards or [],
            "data": data or {},
            "meta": context.get("intent_meta", {})
        }

    # =========================
    # 🧠 TEXT COMPOSER (LIMIT RESTORED)
    # =========================
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
    def _build_visual_explanation(self, outfit, context):

        items = outfit.get("items", []) if isinstance(outfit, dict) else []

        if not items:
            return "Clean and balanced look."

        colors = [i.get("color") for i in items if i]

        parts = []

        if len(set(colors)) == 1:
            parts.append("The monochrome palette keeps it sharp.")
        else:
            parts.append("The colors balance nicely without clashing.")

        if context.get("occasion"):
            parts.append(f"It works well for {context.get('occasion')}.")

        return " ".join(parts)

    # =========================
    # 🧩 HELPERS
    # =========================
    def _safe_actions(self, context):
        try:
            return generate_followup_suggestions(context)
        except Exception:
            return ["Make it sharper", "Change colors", "Try another vibe"]

    def _llm_enabled(self):
        return os.getenv("ENABLE_LLM_SYNTHESIS", "true").lower() in ("1", "true")

    def _proactive_prefix(self, signals):
        if signals.get("suggestion_type") == "morning_outfit":
            return "Here’s something for your morning."
        if signals.get("suggestion_type") == "evening_outfit":
            return "This fits your evening."
        return ""

    def _reaction(self):
        return "This is a strong look."

    def _closer(self):
        return "Want me to tweak it?"

    def _fallback(self, data: dict) -> str:
        if not data:
            return "I couldn’t find anything solid yet."

        chunks = []
        for k, v in data.items():
            if isinstance(v, dict):
                chunks.append(v.get("message") or v.get("summary") or "")
            else:
                chunks.append(str(v))

        return " ".join(chunks)


# singleton
response_assembler = ResponseAssembler()
