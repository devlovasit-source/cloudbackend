from typing import Any, Dict, List

from brain.engines.style_graph_engine import style_graph_engine
from brain.engines.style_rules_engine import style_engine
from brain.engines.palette_engine import palette_engine


class UnifiedStyleScorer:
    """
    🔥 CORE STYLE BRAIN

    Combines:
    - Graph compatibility
    - Palette alignment
    - Style DNA rules
    - Aesthetic balance

    Used by OutfitEngine ONLY
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

        rules = style_engine.get_scoring_rules(
            context.get("style_dna", {}),
            context
        )

        palette = palette_engine.select_palette({
            "event": context.get("occasion"),
            "microtheme": context.get("style_dna", {}).get("aesthetic")
        })

        palette_colors = [c.lower() for c in palette.get("hex", [])]

        score = 0.0

        # =========================
        # 1. GRAPH COMPATIBILITY
        # =========================
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                score += style_graph_engine.pair_weight(
                    graph,
                    items[i].get("id"),
                    items[j].get("id")
                )

        # =========================
        # 2. PALETTE ALIGNMENT
        # =========================
        for item in items:
            color = str(item.get("color", "")).lower()

            if color in palette_colors:
                score += 0.6
            elif self._is_neutral(color):
                score += 0.3

        # =========================
        # 3. STYLE DNA MATCH
        # =========================
        for item in items:
            text = str(item).lower()

            if any(k in text for k in rules.get("preferred_keywords", [])):
                score += 0.5

            if item.get("color", "").lower() in rules.get("preferred_colors", []):
                score += 0.4

            if item.get("type", "").lower() in rules.get("avoided_items", []):
                score -= 1.5

        # =========================
        # 4. AESTHETIC BALANCE
        # =========================
        score += self._aesthetic_score(items)

        return round(score, 3)

    # =========================
    # AESTHETIC LOGIC
    # =========================
    def _aesthetic_score(self, items):

        score = 0.0

        colors = [i.get("color", "").lower() for i in items if i.get("color")]
        fits = [i.get("fit", "").lower() for i in items if i.get("fit")]

        # 🎨 COLOR STORY
        unique_colors = len(set(colors))

        if unique_colors == 1:
            score += 0.8  # monochrome
        elif unique_colors == 2:
            score += 0.6  # balanced
        elif unique_colors >= 3:
            score += 0.4  # contrast

        # 🧍 SILHOUETTE BALANCE
        if self._has_balance(fits):
            score += 0.7

        return score

    # =========================
    # HELPERS
    # =========================
    def _is_neutral(self, color: str) -> bool:
        return color in ["black", "white", "grey", "beige", "navy"]

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
