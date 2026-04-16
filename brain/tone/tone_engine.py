import os
import json


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
    # MAIN ENTRY
    # =========================
    def apply(self, text: str, user_profile: dict = None, signals: dict = None, context: dict = None):

        if not text:
            return text

        user_profile = user_profile or {}
        signals = signals or {}
        context = context or {}

        user_memory = user_profile.get("memory", {})

        # -------------------------
        # 1. BASE DETECTION
        # -------------------------
        generation = self._detect_generation(user_profile)
        context_mode = signals.get("context_mode", "general")
        emotion = signals.get("emotion_state", "neutral")

        context_rules = self.config.get("context_modes", {}).get(context_mode, {})
        emotion_rules = self.config.get("emotion_overrides", {}).get(emotion, {})

        # -------------------------
        # 2. OUTFIT AESTHETIC
        # -------------------------
        aesthetic = context.get("aesthetic") or self._extract_outfit_aesthetic(context)

        # -------------------------
        # 3. LEARNED USER TONE
        # -------------------------
        learned_tone = user_memory.get("tone_preferences", {})

        # -------------------------
        # 4. APPLY BASE CONSTRAINTS
        # -------------------------
        text = self._apply_constraints(text, context_rules, emotion_rules)

        # -------------------------
        # 5. APPLY OUTFIT TONE
        # -------------------------
        text = self._apply_outfit_tone(text, aesthetic)

        # -------------------------
        # 6. APPLY LEARNED USER STYLE
        # -------------------------
        text = self._apply_user_preference(text, learned_tone)

        # -------------------------
        # 7. UPDATE MEMORY (FEEDBACK LOOP)
        # -------------------------
        updated_memory = self._update_learning(user_memory, signals, aesthetic)

        user_profile["memory"] = updated_memory

        return text

    # =========================
    # 🔥 FEEDBACK LEARNING
    # =========================
    def _update_learning(self, memory, signals, aesthetic):

        memory = memory or {}
        prefs = memory.get("tone_preferences", {
            "energy": "balanced",
            "style": "neutral"
        })

        feedback = signals.get("feedback")
        engagement = signals.get("engagement_level")

        if not aesthetic:
            return memory

        # -------------------------
        # POSITIVE SIGNAL
        # -------------------------
        if feedback == "like" or engagement == "high":

            if aesthetic.get("energy") == "bold":
                prefs["energy"] = "bold"

            if aesthetic.get("vibe") == "minimal":
                prefs["style"] = "minimal"

            if aesthetic.get("vibe") == "street":
                prefs["style"] = "expressive"

        # -------------------------
        # NEGATIVE SIGNAL
        # -------------------------
        if feedback == "dislike":

            if aesthetic.get("energy") == "bold":
                prefs["energy"] = "soft"

        memory["tone_preferences"] = prefs
        return memory

    # =========================
    # 👤 USER STYLE APPLY
    # =========================
    def _apply_user_preference(self, text, prefs):

        if not prefs:
            return text

        # ENERGY
        if prefs.get("energy") == "bold":
            text = text.replace("nice", "strong")
            text += " This hits."

        elif prefs.get("energy") == "soft":
            text = text.replace("strong", "easy")
            text += " Feels effortless."

        # STYLE
        if prefs.get("style") == "minimal":
            text = text.replace("Try adding", "You could add")

        elif prefs.get("style") == "expressive":
            text += " This has personality."

        return text

    # =========================
    # 🎨 OUTFIT AWARENESS
    # =========================
    def _extract_outfit_aesthetic(self, context):

        outfit = context.get("outfit_data", {}) or {}
        items = outfit.get("items", [])

        colors = [str(i.get("color", "")).lower() for i in items]
        styles = [str(i.get("style", "")).lower() for i in items]

        dark = {"black", "navy", "charcoal"}
        light = {"white", "beige", "pastel"}

        dark_score = sum(1 for c in colors if c in dark)
        light_score = sum(1 for c in colors if c in light)

        return {
            "energy": "bold" if dark_score > light_score else "soft",
            "vibe": "street" if "street" in styles else "minimal",
            "structure": "sharp" if "formal" in styles else "relaxed"
        }

    def _apply_outfit_tone(self, text, aesthetic):

        if not aesthetic:
            return text

        if aesthetic.get("structure") == "sharp":
            text = text.replace("This works", "This is clean")

        if aesthetic.get("vibe") == "street":
            text += " Lowkey fire."

        if aesthetic.get("vibe") == "minimal":
            text += " Super clean."

        return text

    # =========================
    # BASE RULES
    # =========================
    def _apply_constraints(self, text, context_rules, emotion_rules):

        text = text.replace("!!", "!")

        if emotion_rules.get("sentence_style") == "soft":
            text = text.replace("!", ".")

        if context_rules.get("slang_cap", 0) == 0:
            text = self._remove_slang(text)

        return text

    def _remove_slang(self, text):
        slang_list = self.config.get("slang_libraries", {}).get("gen_z", {}).get("approved_tokens", [])
        for s in slang_list:
            text = text.replace(s, "")
        return text.strip()

    # =========================
    # GENERATION
    # =========================
    def _detect_generation(self, user_profile):

        if not user_profile or not user_profile.get("dob_iso"):
            return "other"

        try:
            year = int(str(user_profile["dob_iso"]).split("-")[0])
        except Exception:
            return "other"

        buckets = self.config.get("generation_buckets", {})

        for name, r in buckets.items():
            if r["dob_year_min"] <= year <= r["dob_year_max"]:
                return name

        return "other"


# Singleton
tone_engine = ToneEngine()
