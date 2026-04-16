from typing import Any, Dict, List

from brain.engines.style_graph_engine import style_graph_engine
from brain.engines.style_rules_engine import style_engine
from brain.engines.palette_engine import palette_engine
from brain.engines.color_normalizer import color_normalizer


class UnifiedStyleScorer:
    """
    🔥 ELITE STYLE BRAIN

    Combines:
    - Graph compatibility
    - Palette intelligence (hex → normalized)
    - Style DNA (personalization)
    - Tone harmony (warm/cool)
    - Aesthetic balance (color + silhouette)
    """

    # =========================
    # MAIN ENTRY
    # =========================
    def score_outfit(
        self,
        items: List[Dict[str, Any]],
        context: Dict[str, Any],
        graph: Dict[str, Any],
    ) -> float:

        if not items:
            return 0.0

        style_dna = context.get("style_dna", {}) or {}

        rules = style_engine.get_scoring_rules(style_dna, context)

        palette = palette_engine.select_palette({
            "event": context.get("occasion"),
            "microtheme": style_dna.get("aesthetic")
        })

        # 🔥 normalize palette colors
        palette_colors = [
            color_normalizer.normalize(c)
            for c in palette.get("hex", [])
        ]

        score = 0.0

        # =========================
        # 1. GRAPH COMPATIBILITY
        # =========================
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a_id = items[i].get("id")
                b_id = items[j].get("id")

                if not a_id or not b_id:
                    continue

                score += style_graph_engine.pair_weight(graph, a_id, b_id)

        # =========================
        # 2. PALETTE + COLOR MATCH
        # =========================
        tones = []

        for item in items:
            color = color_normalizer.normalize(item.get("color"))
            tone = color_normalizer.detect_tone(item.get("color"))

            if not color:
                continue

            tones.append(tone)

            if color in palette_colors:
                score += 0.8  # stronger now
            elif self._is_neutral(color):
                score += 0.4

        # =========================
        # 3. TONE HARMONY (NEW 🔥)
        # =========================
        score += self._tone_harmony_score(tones)

        # =========================
        # 4. STYLE RULES
        # =========================
        for item in items:
            text = str(item).lower()
            color = color_normalizer.normalize(item.get("color"))
            item_type = str(item.get("type", "")).lower()

            if any(k in text for k in rules.get("preferred_keywords", [])):
                score += 0.5

            if color in rules.get("preferred_colors", []):
                score += 0.6

            if item_type in rules.get("avoided_items", []):
                score -= 2.0

        # =========================
        # 5. STYLE DNA (STRONGER)
        # =========================
        for item in items:
            score += self._dna_score(item, style_dna)

        # =========================
        # 6. AESTHETIC BALANCE
        # =========================
        score += self._aesthetic_score(items)

        return round(score, 3)

    # =========================
    # DNA SCORING
    # =========================
    def _dna_score(self, item: Dict[str, Any], dna: Dict[str, Any]) -> float:
        score = 0.0

        color = color_normalizer.normalize(item.get("color"))
        fabric = str(item.get("fabric", "")).lower()
        style = str(item.get("style", "")).lower()
        item_type = str(item.get("type") or item.get("category") or "").lower()

        if color in dna.get("preferred_colors", []):
            score += 1.2

        if fabric in dna.get("preferred_fabrics", []):
            score += 0.6

        if style in dna.get("preferred_styles", []):
            score += 1.0

        if item_type in dna.get("preferred_types", []):
            score += 0.8

        if item_type in dna.get("disliked_items", []):
            score -= 2.0

        return score

    # =========================
    # TONE HARMONY
    # =========================
    def _tone_harmony_score(self, tones: List[str]) -> float:
        if not tones:
            return 0.0

        unique = set(tones)

        # all same → cohesive
        if len(unique) == 1:
            return 0.8

        # mix of warm + neutral OR cool + neutral → stylish
        if "neutral" in unique:
            return 0.5

        # clash (warm + cool)
        if "warm" in unique and "cool" in unique:
            return 0.2

        return 0.4

    # =========================
    # AESTHETIC LOGIC
    # =========================
    def _aesthetic_score(self, items: List[Dict[str, Any]]) -> float:
        score = 0.0

        colors = [
            color_normalizer.normalize(i.get("color"))
            for i in items if i.get("color")
        ]

        fits = [
            str(i.get("fit", "")).lower()
            for i in items if i.get("fit")
        ]

        # 🎨 COLOR STORY
        unique_colors = len(set(colors))

        if unique_colors == 1:
            score += 1.0  # strong monochrome
        elif unique_colors == 2:
            score += 0.7
        elif unique_colors >= 3:
            score += 0.5

        # 🧍 SILHOUETTE BALANCE
        if self._has_balance(fits):
            score += 0.8

        return score

    # =========================
    # HELPERS
    # =========================
    def _is_neutral(self, color: str) -> bool:
        return color in ["black", "white", "grey", "gray", "beige", "navy", "cream"]

    def _has_balance(self, fits: List[str]) -> bool:
        combos = [
            ("slim", "relaxed"),
            ("oversized", "slim"),
        ]

        for f1 in fits:
            for f2 in fits:
                if f1 == f2:
                    continue

                for a, b in combos:
                    if (a in f1 and b in f2) or (a in f2 and b in f1):
                        return True

        return False


# Singleton
style_scorer = UnifiedStyleScorer()
