import json
import logging
import os
from typing import Dict, Any, List

from brain.tone.tone_engine import tone_engine
from services.ai_gateway import chat_completion

logger = logging.getLogger("ahvi.response_assembler")


class ResponseAssembler:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        profile_path = os.path.join(base_dir, "config", "assembly_profiles.json")

        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                self.config = json.load(f).get("assembly_profiles", {})
        except Exception as exc:
            logger.warning("Assembly profile load failed: %s", exc)
            self.config = {}

    # =========================
    # 🔥 MAIN ENTRY
    # =========================
    def assemble(self, merged_output: dict, context: dict = None) -> str:
        context = context or {}
        user_profile = context.get("user_profile", {})

        # -------------------------
        # 🔥 STYLING MODE (CORE)
        # -------------------------
        if merged_output.get("type") == "styling":
            return self._assemble_styling(merged_output, context)

        # -------------------------
        # DEFAULT FLOW
        # -------------------------
        data = merged_output.get("data", {}) if isinstance(merged_output, dict) else {}
        domains = context.get("domains") or list(data.keys())

        parts = [self._reaction(len(domains) > 1)]

        intelligence = (
            merged_output.get("message")
            or self._fallback_synthesis(data, domains)
        )

        if intelligence:
            parts.append(intelligence)

        parts.append(self._closer())

        final_text = self._apply_global_rules(parts)

        return tone_engine.apply(
            final_text,
            user_profile=user_profile,
            signals=context.get("signals"),
        )

    # =========================
    # 🔥 STYLING INTELLIGENCE
    # =========================
    def _assemble_styling(self, merged_output: dict, context: dict) -> str:
        outfits = merged_output.get("outfits", [])
        user_profile = context.get("user_profile", {})

        if not outfits:
            return "Nothing solid here yet — let's refine your wardrobe."

        best = outfits[0]
        items = best.get("items", [])

        explanation = self._build_visual_explanation(items, context)

        # hint if multiple outfits exist
        if len(outfits) > 1:
            explanation += " I have a couple of other options too if you want to explore."

        final_text = self._apply_global_rules([
            self._reaction(False),
            explanation
        ])

        return tone_engine.apply(
            final_text,
            user_profile=user_profile,
            signals=context.get("signals"),
        )

    # =========================
    # 🔥 VISUAL REASONING CORE
    # =========================
    def _build_visual_explanation(self, items: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
        if not items:
            return "This is a clean and balanced look."

        colors = [i.get("color", "").lower() for i in items if i]
        categories = [i.get("category", "").lower() for i in items if i]
        styles = [i.get("style", "").lower() for i in items if i]

        parts = []

        # -------------------------
        # 🎨 COLOR LOGIC
        # -------------------------
        unique_colors = list(set(colors))

        if len(unique_colors) == 1:
            parts.append("The monochrome palette keeps it sharp and intentional.")
        elif len(unique_colors) <= 3:
            parts.append("The colors balance well without clashing.")

        # -------------------------
        # 🧍 SILHOUETTE
        # -------------------------
        if "top" in categories and "bottom" in categories:
            parts.append("The top–bottom balance keeps the silhouette structured.")

        if "outerwear" in categories:
            parts.append("The layering adds depth without overwhelming the look.")

        # -------------------------
        # 🧠 STYLE CONSISTENCY
        # -------------------------
        if len(set(styles)) == 1 and styles[0]:
            parts.append(f"It stays consistent with a {styles[0]} aesthetic.")

        # -------------------------
        # 🎯 CONTEXT FIT
        # -------------------------
        occasion = context.get("occasion")
        weather = context.get("weather")

        if occasion:
            parts.append(f"It fits the {occasion} setting naturally.")

        if weather == "summer":
            parts.append("It feels breathable and light for the heat.")
        elif weather == "winter":
            parts.append("The structure works well for layering in colder weather.")
        elif weather == "rainy":
            parts.append("It stays practical for unpredictable weather.")

        # -------------------------
        # 🔥 FALLBACK
        # -------------------------
        if not parts:
            parts.append("Everything here just works — clean, balanced, and effortless.")

        return " ".join(parts)

    # =========================
    # MULTI DOMAIN SYNTHESIS
    # =========================
    def _synthesize(self, data: dict, domains: list, user_profile: dict) -> str:
        llm_enabled = os.getenv("ENABLE_LLM_SYNTHESIS", "false").lower() in ("1", "true", "yes")
        if not llm_enabled:
            return self._fallback_synthesis(data, domains)

        system_prompt = (
            "You are AHVI. Combine results into a sharp, structured response. "
            "Avoid generic phrasing. Keep it tight and useful."
        )

        prompt = (
            f"Domains: {', '.join(domains)}\n"
            f"User Profile: {json.dumps(user_profile)}\n"
            f"Engine Data: {json.dumps(data)}\n"
            "Write a clean, confident response."
        )

        try:
            return chat_completion(
                [{"role": "user", "content": prompt}],
                system_instruction=system_prompt,
            )
        except Exception as exc:
            logger.warning("LLM synthesis failed: %s", exc)
            return self._fallback_synthesis(data, domains)

    # =========================
    # FALLBACK
    # =========================
    def _fallback_synthesis(self, data: dict, domains: list) -> str:
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

    # =========================
    # UX LAYERS
    # =========================
    def _reaction(self, is_multi_intent: bool) -> str:
        return (
            "Alright, I pulled everything together."
            if is_multi_intent
            else "This is a strong look."
        )

    def _closer(self) -> str:
        return "Want me to refine this further?"

    def _apply_global_rules(self, parts: list) -> str:
        rules = self.config.get("global_rules", {})
        max_q = rules.get("max_questions_per_response", 1)
        max_sent = rules.get("max_sentences_layer_1", 3)

        cleaned = []
        question_count = 0

        for part in parts:
            if not part:
                continue

            if "?" in part:
                if question_count >= max_q:
                    continue
                question_count += 1

            cleaned.append(part)

        final = "\n\n".join(cleaned)

        sentences = final.split(". ")
        if len(sentences) > max_sent:
            final = ". ".join(sentences[:max_sent])

        return final.strip()


# Singleton
response_assembler = ResponseAssembler()
