from typing import Dict, List
import random


class StyleLanguageEngine:
    """
    🔥 STYLE LANGUAGE ENGINE

    Converts structured outfit → editorial sentence with:
    - aesthetic-aware tone
    - occasion-aware phrasing
    - micro-variation (non-repetitive)
    - dynamic outfit composition (not fixed categories)
    """

    # =========================
    # ITEM → TEXT
    # =========================
    def item_to_text(self, item: Dict) -> str:

        if not item:
            return ""

        color = self._clean(item.get("color"))
        fit = self._clean(item.get("fit"))
        style = self._clean(item.get("style"))
        fabric = self._clean(item.get("fabric"))
        item_type = self._clean(item.get("type") or item.get("category"))

        parts = []

        if fit:
            parts.append(fit)

        if color:
            parts.append(color)

        if fabric:
            parts.append(fabric)

        if style and style not in parts:
            parts.append(style)

        if item_type:
            parts.append(item_type)

        return " ".join(parts).strip()

    # =========================
    # 🔥 OUTFIT → SENTENCE
    # =========================
    def outfit_to_sentence(self, items: List[Dict], context: Dict = None) -> str:

        if not items:
            return ""

        context = context or {}
        style_dna = context.get("style_dna", {}) or {}

        aesthetic = (
            style_dna.get("primary_aesthetic")
            or context.get("aesthetic")
            or "modern"
        ).lower()

        occasion = str(context.get("occasion", "")).lower()

        core, layers, extras = [], [], []

        for item in items:
            text = self.item_to_text(item)
            category = str(item.get("category", "")).lower()

            if not text:
                continue

            if category in ["top", "shirt", "tshirt", "dress"]:
                core.append(text)

            elif category in ["bottom", "pants", "jeans", "trousers"]:
                core.append(text)

            elif category in ["outerwear", "jacket", "blazer"]:
                layers.append(text)

            else:
                extras.append(text)

        parts = []

        if core:
            parts.append(", ".join(core))

        if layers:
            parts.append(f"{self._layer_connector()} {', '.join(layers)}")

        if extras:
            parts.append(f"{self._finish_connector()} {', '.join(extras)}")

        base = ", ".join(parts)

        # =========================
        # 🔥 TONE LAYERS
        # =========================
        opener = self._pick_opener(aesthetic)
        occasion_phrase = self._occasion_phrase(occasion)

        return f"{opener} {base}. {occasion_phrase}".strip()

    # =========================
    # 🔥 OCCASION ENGINE
    # =========================
    def _occasion_phrase(self, occasion: str) -> str:

        occasion_map = {
            "date": [
                "Feels confident without trying too hard.",
                "Strikes the right balance between sharp and effortless.",
            ],
            "dinner": [
                "Clean, confident, and just elevated enough for the setting.",
                "Refined without feeling overdone.",
            ],
            "office": [
                "Polished enough for work, without feeling stiff.",
                "Keeps things sharp while staying natural.",
            ],
            "work": [
                "Structured and appropriate, but still easy to wear.",
            ],
            "travel": [
                "Comfortable, practical, and still put-together.",
                "Easy to move in without losing structure.",
            ],
            "airport": [
                "Effortless and travel-ready, without looking lazy.",
            ],
            "party": [
                "It stands out without trying too hard.",
                "Strong presence, but still controlled.",
            ],
            "vacation": [
                "Relaxed, breathable, and easygoing.",
                "Feels light and effortless for the setting.",
            ],
        }

        for key, phrases in occasion_map.items():
            if key in occasion:
                return random.choice(phrases)

        return random.choice([
            "Everything comes together effortlessly.",
            "It all feels balanced and intentional.",
        ])

    # =========================
    # 🔥 AESTHETIC OPENERS
    # =========================
    def _pick_opener(self, aesthetic: str) -> str:

        tone_map = {
            "minimal": [
                "Clean and sharp —",
                "Stripped-back and refined —",
                "Quietly confident —",
            ],
            "street": [
                "Relaxed and expressive —",
                "Easygoing with edge —",
                "Off-duty and confident —",
            ],
            "luxury": [
                "Structured and elevated —",
                "Polished with presence —",
                "Refined and intentional —",
            ],
            "formal": [
                "Structured and composed —",
                "Sharp and well-defined —",
            ],
            "casual": [
                "Effortless and easy —",
                "Relaxed but put-together —",
                "Simple, done right —",
            ],
        }

        for key, options in tone_map.items():
            if key in aesthetic:
                return random.choice(options)

        return random.choice([
            "Well-balanced and considered —",
            "Cleanly put together —",
            "Sharp without trying too hard —",
        ])

    # =========================
    # CONNECTORS (VARIATION)
    # =========================
    def _layer_connector(self) -> str:
        return random.choice([
            "layered with",
            "topped with",
            "finished with a layer of",
        ])

    def _finish_connector(self) -> str:
        return random.choice([
            "finished with",
            "grounded by",
            "anchored with",
        ])

    # =========================
    # CLEANER
    # =========================
    def _clean(self, value):

        if not value:
            return ""

        v = str(value).strip().lower()

        blacklist = ["none", "unknown", "null", ""]

        if v in blacklist:
            return ""

        return v


# Singleton
style_language_engine = StyleLanguageEngine()
