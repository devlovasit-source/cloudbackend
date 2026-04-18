from typing import Any, Dict, List
import random

from brain.engines.style_graph_engine import style_graph_engine
from brain.engines.style_rules_engine import style_engine
from brain.engines.styling.palette_engine import palette_engine
from brain.engines.color_normalizer import color_normalizer

from services.qdrant_service import qdrant_service
from services.embedding_service import encode_metadata
from services.image_embedding_service import encode_image_url


class UnifiedStyleScorer:
    """
    🔥 ELITE STYLE SCORER

    Adds:
    - refinement awareness
    - stronger style DNA influence
    - memory signals fusion (explicit + embedding)
    - wardrobe-aware bias
    - stability controls
    """

    def score_outfit(
        self,
        items: List[Dict[str, Any]],
        context: Dict[str, Any],
        graph: Dict[str, Any],
    ) -> Dict[str, Any]:

        if not items:
            return {"score": 0.0, "label": "Weak", "reasons": []}

        style_dna = context.get("style_dna", {}) or {}
        memory = context.get("user_memory", {}) or {}
        refinement = context.get("refinement")

        confidence = float(style_dna.get("confidence", 0.5))
        exploration_factor = max(0.0, 1.0 - confidence)

        reasons = []
        score = 0.0

        # =========================
        # RULES + PALETTE
        # =========================
        rules = style_engine.get_scoring_rules(style_dna, context)

        palette = palette_engine.select_palette({
            "event": context.get("occasion"),
            "microtheme": style_dna.get("primary_aesthetic")
        })

        palette_colors = [
            color_normalizer.normalize(c)
            for c in palette.get("hex", [])
        ]

        # =========================
        # 1. GRAPH COMPATIBILITY
        # =========================
        graph_score = 0.0

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a_id = items[i].get("id")
                b_id = items[j].get("id")

                if a_id and b_id:
                    graph_score += style_graph_engine.pair_weight(graph, a_id, b_id)

        score += graph_score
        if graph_score > 1:
            reasons.append("items pair well together")

        # =========================
        # 2. COLOR + RULES
        # =========================
        for item in items:
            color = color_normalizer.normalize(item.get("color"))
            item_type = str(item.get("type", "")).lower()

            if color in palette_colors:
                score += 1.0
                reasons.append("palette aligned")

            elif self._is_neutral(color):
                score += 0.4

            if color in rules.get("preferred_colors", []):
                score += 0.6

            if item_type in rules.get("avoided_items", []):
                score -= 2.0
                reasons.append("conflicts with style")

        # =========================
        # 3. AESTHETIC BALANCE
        # =========================
        aesthetic_score = self._aesthetic_score(items)
        score += aesthetic_score

        if aesthetic_score > 0.7:
            reasons.append("clean aesthetic balance")

        # =========================
        # 4. 🔥 STYLE DNA (STRONG)
        # =========================
        dna_score = self._dna_score(items, style_dna)
        score += dna_score * (0.5 + confidence)

        if dna_score > 0:
            reasons.append("matches your style")

        # =========================
        # 5. 🔥 MEMORY (EXPLICIT)
        # =========================
        memory_score = self._memory_signal_score(items, memory)
        score += memory_score * 1.2

        if memory_score > 0:
            reasons.append("aligned with your past choices")

        # =========================
        # 6. 🔥 EMBEDDING MEMORY
        # =========================
        vector = self._build_outfit_embedding(items)

        if vector and len(vector) > 100:
            qdrant_score = self._qdrant_score(context.get("user_id"), vector)
            score += qdrant_score * (0.4 + confidence)

        # =========================
        # 7. 🔥 REFINEMENT BOOST
        # =========================
        if refinement:
            refine_score = self._refinement_score(items, refinement)
            score += refine_score

            if refine_score > 0:
                reasons.append(f"refined for {refinement}")

        # =========================
        # 8. EXPLORATION
        # =========================
        score += self._exploration_boost(items, exploration_factor)

        # =========================
        # FINAL LABEL
        # =========================
        label = self._label(score)

        return {
            "score": round(score, 3),
            "label": label,
            "reasons": list(set(reasons))[:3]
        }

    # =========================
    # 🔥 DNA SCORING
    # =========================
    def _dna_score(self, items, dna):

        if not dna:
            return 0

        score = 0

        preferred_styles = dna.get("preferred_styles", [])
        preferred_colors = dna.get("preferred_colors", [])

        for i in items:
            style = str(i.get("style", "")).lower()
            color = str(i.get("color", "")).lower()

            if style in preferred_styles:
                score += 0.6

            if color in preferred_colors:
                score += 0.5

        return score

    # =========================
    # 🔥 MEMORY SIGNAL SCORE
    # =========================
    def _memory_signal_score(self, items, memory):

        signals = memory.get("memory_signals", {})
        if not signals:
            return 0

        score = 0

        for item in items:
            style = str(item.get("style", "")).lower()
            color = str(item.get("color", "")).lower()

            if style in signals.get("preferred_styles", []):
                score += 0.5

            if color in signals.get("liked_colors", []):
                score += 0.4

        return score

    # =========================
    # 🔥 REFINEMENT SCORE
    # =========================
    def _refinement_score(self, items, refinement):

        score = 0

        for item in items:
            style = str(item.get("style", "")).lower()

            if refinement == "sharp" and style in ["formal", "structured"]:
                score += 0.5

            if refinement == "relaxed" and style in ["casual", "loose"]:
                score += 0.5

            if refinement == "minimal" and item.get("pattern") == "solid":
                score += 0.4

        return score

    # =========================
    # EMBEDDING
    # =========================
    def _build_outfit_embedding(self, items):

        text = " ".join([
            f"{i.get('type','')} {i.get('color','')} {i.get('style','')}"
            for i in items
        ])

        try:
            text_vector = encode_metadata({"text": text}) or []
        except Exception:
            text_vector = []

        return text_vector

    # =========================
    # QDRANT
    # =========================
    def _qdrant_score(self, user_id, vector):

        if not user_id:
            return 0

        try:
            liked = qdrant_service.search_user_memory(
                user_id=user_id,
                vector=vector,
                memory_type="liked",
                limit=3
            )

            disliked = qdrant_service.search_user_memory(
                user_id=user_id,
                vector=vector,
                memory_type="disliked",
                limit=3
            )

            like_score = sum(r.get("score", 0) for r in liked)
            dislike_score = sum(r.get("score", 0) for r in disliked)

            return (like_score * 1.2) - (dislike_score * 1.5)

        except Exception:
            return 0

    # =========================
    # EXPLORATION
    # =========================
    def _exploration_boost(self, items, factor):

        if factor <= 0:
            return 0

        colors = [i.get("color") for i in items if i.get("color")]
        styles = [i.get("style") for i in items if i.get("style")]

        score = 0

        if len(set(colors)) >= 3:
            score += 0.5 * factor

        if len(set(styles)) >= 2:
            score += 0.4 * factor

        score += random.uniform(0, 0.2) * factor

        return score

    # =========================
    # AESTHETIC
    # =========================
    def _aesthetic_score(self, items):

        colors = [
            color_normalizer.normalize(i.get("color"))
            for i in items if i.get("color")
        ]

        unique = len(set(colors))

        if unique == 1:
            return 1.0
        elif unique == 2:
            return 0.7
        elif unique >= 3:
            return 0.5

        return 0

    # =========================
    # LABEL
    # =========================
    def _label(self, score):

        if score >= 6:
            return "Excellent"
        if score >= 4:
            return "Strong"
        if score >= 2:
            return "Good"
        return "Basic"

    def _is_neutral(self, color):
        return color in ["black", "white", "grey", "gray", "beige", "navy", "cream"]


# singleton
style_scorer = UnifiedStyleScorer()
