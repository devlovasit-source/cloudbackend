
import json
import logging
import os
from typing import Dict, Any, List

from brain.tone.tone_engine import tone_engine
from services.llm_service import (
    generate_outfit_explanation,
    generate_style_advice,
    generate_followup_suggestions,
)
from services.ai_gateway import chat_completion

logger = logging.getLogger("ahvi.response_assembler")


class ResponseAssembler:

    # =========================
    # MAIN ENTRY
    # =========================
    def assemble(self, merged_output: dict, context: dict = None):
        context = context or {}
        user_profile = context.get("user_profile", {})
        response_type = merged_output.get("type")

        # 🔥 PRIMARY DOMAIN
        if response_type == "styling":
            return self._assemble_styling(merged_output, context)

        # 🧠 SECONDARY DOMAINS
        return self._assemble_multi_domain(merged_output, context, user_profile)

    # =========================
    # 🔥 STYLING (PREMIUM)
    # =========================
    def _assemble_styling(self, merged_output: dict, context: dict):

        data = merged_output.get("data", {})
        outfits = data.get("outfits", [])
        boards = data.get("boards", [])

        if not outfits:
            return "Nothing solid here yet — let's refine your wardrobe."

        user_profile = context.get("user_profile", {})
        style_dna = context.get("style_dna", {})
        refinement = context.get("refinement")
        signals = context.get("proactive_signals", {})

        best = outfits[0]
        items = best.get("items", [])

        # -------------------------
        # BASE (RULE)
        # -------------------------
        base = self._build_visual_explanation(items, context)

        # -------------------------
        # 🔥 LLM ENHANCEMENT
        # -------------------------
        if self._llm_enabled():
            try:
                llm_text = generate_outfit_explanation(
                    outfits=outfits,
                    context={
                        "occasion": context.get("occasion"),
                        "weather": context.get("weather"),
                        "refinement": refinement,
                    },
                    user_profile=user_profile,
                    style_dna=style_dna,
                    signals=context.get("signals"),
                )

                if llm_text and llm_text.strip():
                    base = llm_text

            except Exception as e:
                logger.warning("LLM styling failed: %s", e)

        # -------------------------
        # PROACTIVE PREFIX
        # -------------------------
        prefix = self._build_proactive_prefix(signals)

        message = self._apply_global_rules([
            prefix,
            self._reaction(False),
            base
        ])

        message = tone_engine.apply(
            message,
            user_profile=user_profile,
            signals=context.get("signals"),
        )

        # -------------------------
        # 🔥 ACTIONS
        # -------------------------
        actions = self._safe_actions(context, outfits)

        return {
            "message": message,
            "actions": actions
        }

    # =========================
    # 🧠 MULTI-DOMAIN (CLEAN)
    # =========================
    def _assemble_multi_domain(self, merged_output, context, user_profile):

        data = merged_output.get("data", {}) if isinstance(merged_output, dict) else {}
        domains = context.get("domains") or list(data.keys())

        base_text = (
            merged_output.get("message")
            or self._fallback_synthesis(data, domains)
        )

        # 🔥 OPTIONAL LLM POLISH
        if self._llm_enabled() and base_text:
            try:
                improved = generate_style_advice(
                    user_input=base_text,
                    wardrobe_summary=str(context.get("wardrobe", "")),
                    user_profile=user_profile,
                    signals=context.get("signals"),
                )

                if improved and improved.strip():
                    base_text = improved

            except Exception as e:
                logger.warning("LLM multi-domain failed: %s", e)

        final = self._apply_global_rules([
            self._reaction(len(domains) > 1),
            base_text,
            self._closer()
        ])

        return tone_engine.apply(
            final,
            user_profile=user_profile,
            signals=context.get("signals"),
        )

    # =========================
    # 🔥 VISUAL LOGIC
    # =========================
    def _build_visual_explanation(self, items, context):

        if not items:
            return "This is a clean and balanced look."

        colors = [i.get("color", "") for i in items if i]
        categories = [i.get("category", "") for i in items if i]
        styles = [i.get("style", "") for i in items if i]

        parts = []

        if len(set(colors)) == 1:
            parts.append("The monochrome palette keeps it sharp and intentional.")
        else:
            parts.append("The colors stay balanced without clashing.")

        if "top" in categories and "bottom" in categories:
            parts.append("The silhouette feels structured and clean.")

        if len(set(styles)) == 1 and styles[0]:
            parts.append(f"It stays consistent with a {styles[0]} aesthetic.")

        if context.get("occasion"):
            parts.append(f"It fits the {context.get('occasion')} setting naturally.")

        return " ".join(parts)

    # =========================
    # 🔥 HELPERS
    # =========================
    def _safe_actions(self, context, outfits):
        try:
            return generate_followup_suggestions(context, outfits)
        except Exception:
            return ["Make it sharper", "Try a relaxed version"]

    def _llm_enabled(self):
        return os.getenv("ENABLE_LLM_SYNTHESIS", "false").lower() in ("1", "true", "yes")

    def _build_proactive_prefix(self, signals):
        if not signals:
            return ""

        if signals.get("suggestion_type") == "morning_outfit":
            return "Here’s something that works well for your morning."

        if signals.get("suggestion_type") == "evening_outfit":
            return "This would fit nicely for your evening."

        return ""

    def _reaction(self, multi):
        return (
            "Alright, I pulled everything together."
            if multi else "This is a strong look."
        )

    def _closer(self):
        return "Want me to refine this further?"

    def _apply_global_rules(self, parts):

        parts = [p for p in parts if p]

        text = "\n\n".join(parts)

        sentences = text.split(". ")
        if len(sentences) > 3:
            text = ". ".join(sentences[:3])

        return text.strip()

    # =========================
    # FALLBACK
    # =========================
    def _fallback_synthesis(self, data: dict, domains: list):

        if not data:
            return ""

        chunks = []

        for domain in domains:
            domain_data = data.get(domain, {})

            if isinstance(domain_data, dict):
                msg = domain_data.get("message") or domain_data.get("summary")
                if not msg:
                    msg = f"Your {domain} plan is ready."
            else:
                msg = str(domain_data)

            chunks.append(msg)

        return "\n\n".join(chunks)


# Singleton
response_assembler = ResponseAssembler()
