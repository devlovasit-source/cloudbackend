from typing import Any, Dict, List

from brain.engines.style_graph_engine import style_graph_engine
from brain.engines.style_rules_engine import style_engine
from brain.engines.palette_engine import palette_engine
from brain.engines.color_normalizer import color_normalizer


class UnifiedStyleScorer:
    """
    🔥 ELITE STYLE BRAIN (UPGRADED)

    Combines:
    - Graph compatibility
    - Palette intelligence
    - Multi-aesthetic blending
    - Skin tone compatibility
    - Style DNA personalization
    - Tone harmony
    - Aesthetic balance
    - Diversity control
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

        # 🔥 MULTI-AESTHETIC SETUP
        primary = style_dna.get("primary_aesthetic")
        secondary = style_dna.get("secondary_aesthetics", []) or []

        aesthetic_weights = {}
        if primary:
            aesthetic_weights[primary] = 1.0

        for i, a in enumerate(secondary):
            aesthetic_weights[a] = max(0.4, 0.7 - (i * 0.1))

        # fallback
        if not aesthetic_weights:
            fallback = style_dna.get("aesthetic")
            if fallback:
                aesthetic_weights[fallback] = 1.0

        # 🔥 RULES
        rules = style_engine.get_scoring_rules(style_dna, context)

        # 🔥 PALETTE (use dominant aesthetic)
        dominant = primary or (secondary[0] if secondary else None)

        palette = palette_engine.select_palette({
            "event": context.get("occasion"),
            "microtheme": dominant
        })

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
        # 2. COLOR + PALETTE + SKIN
        # =========================
        tones = []

        skin = context.get("skin_tone_data", {})
        compatible_colors = skin.get("compatible_colors", [])

        for item in items:
            raw_color = item.get("color")

            color = color_normalizer.normalize(raw_color)
            tone = color_normalizer.detect_tone(raw_color) if raw_color else "neutral"

            if not color:
                continue

            tones.append(tone)

            # palette match
            if color in palette_colors:
                score += 0.8
            elif self._is_neutral(color):
                score += 0.4

            # skin tone compatibility
            if color in compatible_colors:
                score += 0.5

        # =========================
        # 3. TONE HARMONY
        # =========================
        score += self._tone_harmony_score(tones)

        # =========================
        # 4. STYLE RULES
        # =========================
        for item in items:
            text = f"{item.get('type','')} {item.get('style','')} {item.get('fabric','')}".lower()
            color = color_normalizer.normalize(item.get("color"))
            item_type = str(item.get("type", "")).lower()

            if any(k in text for k in rules.get("preferred_keywords", [])):
                score += 0.5

            if color in rules.get("preferred_colors", []):
                score += 0.6

            if item_type in rules.get("avoided_items", []):
                score -= 2.0

        # =========================
        # 5. STYLE DNA
        # =========================
        for item in items:
            score += self._dna_score(item, style_dna)

        # =========================
        # 6. MULTI-AESTHETIC MATCH 🔥
        # =========================
        for item in items:
            score += self._aesthetic_match_score(item, aesthetic_weights)

        # =========================
        # 7. AESTHETIC BALANCE
        # =========================
        score += self._aesthetic_score(items)

        # =========================
        # 8. DIVERSITY PENALTY
        # =========================
        score += self._diversity_penalty(items)

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
    # MULTI-AESTHETIC MATCH
    # =========================
    def _aesthetic_match_score(self, item: Dict[str, Any], weights: Dict[str, float]) -> float:
        score = 0.0

        item_style = str(item.get("style", "")).lower()
        item_type = str(item.get("type", "")).lower()

        for aesthetic, weight in weights.items():

            if not aesthetic:
                continue

            a = aesthetic.lower()

            if "minimal" in a and any(k in item_style for k in ["clean", "plain", "tailored"]):
                score += 0.6 * weight

            if "street" in a and any(k in item_style for k in ["graphic", "oversized", "casual"]):
                score += 0.6 * weight

            if "formal" in a and any(k in item_type for k in ["blazer", "trouser", "suit"]):
                score += 0.6 * weight

            if "boho" in a and any(k in item_style for k in ["printed", "layered", "ethnic"]):
                score += 0.6 * weight

        return score

    # =========================
    # TONE HARMONY
    # =========================
    def _tone_harmony_score(self, tones: List[str]) -> float:
        if not tones:
            return 0.0

        unique = set(tones)

        if len(unique) == 1:
            return 0.8

        if "neutral" in unique:
            return 0.5

        if "warm" in unique and "cool" in unique:
            return 0.2

        return 0.4

    # =========================
    # AESTHETIC BALANCE
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

        unique_colors = len(set(colors))

        if unique_colors == 1:
            score += 1.0
        elif unique_colors == 2:
            score += 0.7
        elif unique_colors >= 3:
            score += 0.5

        if self._has_balance(fits):
            score += 0.8

        return score

    # =========================
    # DIVERSITY
    # =========================
    def _diversity_penalty(self, items):
        types = [str(i.get("type", "")).lower() for i in items]

        if len(set(types)) < 2:
            return -0.5

        return 0.0

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
