import os
import json
import random
from typing import Dict, Any, List

from brain.tone.archetype_engine import archetype_engine


class ToneEngine:

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(base_dir, "shared", "tone", "tone_engine.json")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self.config = json.load(f).get("ahvi_tone_engine_v1", {})
        except Exception as e:
            print(f"WARN: Tone engine load failed: {e}")
            self.config = {}

    # =========================
    # MAIN ENTRY (FINAL)
    # =========================
    def apply(
        self,
        text: str,
        user_profile: dict = None,
        signals: dict = None,
        context: dict = None,
        user_memory: dict = None
    ) -> str:

        if not text:
            return text

        user_profile = user_profile or {}
        signals = signals or {}
        context = context or {}
        user_memory = user_memory or {}

        # -------------------------
        # 🔥 BUILD BASE TONE
        # -------------------------
        tone = self._build_tone_profile(user_profile, signals)

        # -------------------------
        # 🔥 GET ARCHETYPE BLEND
        # -------------------------
        archetypes = archetype_engine.select(context, user_memory)

        # -------------------------
        # 🔥 APPLY CONSTRAINTS FIRST
        # -------------------------
        text = self._apply_constraints(text, tone)

        # -------------------------
        # 🔥 APPLY ARCHETYPE BLEND
        # -------------------------
        text = self._apply_archetype_blend(text, archetypes)

        # -------------------------
        # 🔥 FINAL PERSONALITY TOUCH
        # -------------------------
        text = self._apply_personality(text, tone, user_memory)

        return text.strip()

    # =========================
    # TONE PROFILE
    # =========================
    def _build_tone_profile(self, user_profile, signals):

        generation = self._detect_generation(user_profile)
        context_mode = signals.get("context_mode", "general")
        emotion = signals.get("emotion_state", "neutral")

        gen_rules = self.config.get("generation_defaults", {}).get(generation, {})
        ctx_rules = self.config.get("context_modes", {}).get(context_mode, {})
        emo_rules = self.config.get("emotion_overrides", {}).get(emotion, {})

        tone = {
            "slang": gen_rules.get("base_slang", 10),
            "emoji": gen_rules.get("base_emoji", 0),
            "sass": gen_rules.get("base_sass", 10),
        }

        # context caps
        tone["slang"] = min(tone["slang"], ctx_rules.get("slang_cap", tone["slang"]))
        tone["emoji"] = min(tone["emoji"], ctx_rules.get("emoji_cap", tone["emoji"]))

        # emotion overrides (highest priority)
        if emo_rules:
            tone["slang"] = emo_rules.get("slang_cap", tone["slang"])
            tone["emoji"] = emo_rules.get("emoji_cap", tone["emoji"])
            tone["sass"] = emo_rules.get("sass_cap", tone["sass"])

        return tone

    # =========================
    # ARCHETYPE BLENDING
    # =========================
    def _apply_archetype_blend(self, text: str, archetypes: List[Dict]) -> str:

        if not archetypes:
            return text

        final_text = text

        for arc in archetypes:
            arc_type = arc.get("type")
            weight = arc.get("weight", 0)

            config = archetype_engine.get_config(arc_type)
            if not config:
                continue

            speech = config.get("speech", {})
            style = speech.get("style")

            # -------------------------
            # 🔥 STYLE TRANSFORMS
            # -------------------------
            if style == "visual_punchy" and weight > 0.5:
                final_text = self._shorten(final_text)
                final_text = final_text.replace("This works", "This works clean")

            elif style == "casual_fun" and weight > 0.3:
                final_text = "Okay " + final_text.lower()

            elif style == "minimal_polished" and weight > 0.5:
                final_text = final_text.replace("!", ".")

            # -------------------------
            # 🔥 SIGNATURE LINES
            # -------------------------
            signatures = speech.get("signature_lines", [])
            if signatures and weight > 0.4:
                if random.random() < weight:
                    final_text += " " + random.choice(signatures)

        return final_text

    # =========================
    # PERSONALITY LAYER
    # =========================
    def _apply_personality(self, text, tone, user_memory):

        interaction = user_memory.get("interaction_style", {})

        # 🔥 confidence boost
        if tone["sass"] > 15:
            text = text.replace("It works", "This works really well")

        # 🔥 slang (controlled)
        if interaction.get("likes_slang") and tone["slang"] > 20:
            slang_list = self.config.get("slang_libraries", {}).get("gen_z", {}).get("approved_tokens", [])
            if slang_list:
                text += " " + random.choice(slang_list[:3])

        return text

    # =========================
    # CONSTRAINTS
    # =========================
    def _apply_constraints(self, text, tone):

        text = text.replace("!!", "!")

        if tone["emoji"] == 0:
            text = self._remove_emojis(text)

        if tone["slang"] == 0:
            text = self._remove_slang(text)

        return text

    # =========================
    # HELPERS
    # =========================
    def _detect_generation(self, user_profile):
        if not user_profile or not user_profile.get("dob_iso"):
            return "other"

        try:
            year = int(str(user_profile["dob_iso"]).split("-")[0])
        except Exception:
            return "other"

        for name, r in self.config.get("generation_buckets", {}).items():
            if r["dob_year_min"] <= year <= r["dob_year_max"]:
                return name

        return "other"

    def _remove_slang(self, text):
        slang_list = self.config.get("slang_libraries", {}).get("gen_z", {}).get("approved_tokens", [])
        for s in slang_list:
            text = text.replace(s, "")
        return text.strip()

    def _remove_emojis(self, text):
        return text.encode("ascii", "ignore").decode()

    def _shorten(self, text):
        sentences = text.split(". ")
        return ". ".join(sentences[:2])


# Singleton
tone_engine = ToneEngine()
