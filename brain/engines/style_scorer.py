from typing import Any, Dict, List
import random

from brain.engines.style_graph_engine import style_graph_engine
from brain.engines.style_rules_engine import style_engine
from brain.engines.palette_engine import palette_engine
from brain.engines.color_normalizer import color_normalizer

from services.qdrant_service import qdrant_service
from services.embedding_service import encode_metadata
from services.image_embedding_service import encode_image_url  # 🔥 NEW


class UnifiedStyleScorer:
    """
    🔥 ULTIMATE STYLE BRAIN (UPGRADED)

    Adds:
    - DNA confidence adaptivity
    - Qdrant learning
    - Conversation memory
    - Exploration boost
    - 🔥 Visual similarity (image embeddings)
    """

    def score_outfit(
        self,
        items: List[Dict[str, Any]],
        context: Dict[str, Any],
        graph: Dict[str, Any],
    ) -> float:

        if not items:
            return 0.0

        style_dna = context.get("style_dna", {}) or {}
        confidence = float(style_dna.get("confidence", 0.5))

        exploration_factor = max(0.0, 1.0 - confidence)

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

        score = 0.0

        # =========================
        # 1. GRAPH
        # =========================
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a_id = items[i].get("id")
                b_id = items[j].get("id")

                if a_id and b_id:
                    score += style_graph_engine.pair_weight(graph, a_id, b_id)

        # =========================
        # 2. COLOR + RULES
        # =========================
        for item in items:
            color = color_normalizer.normalize(item.get("color"))
            item_type = str(item.get("type", "")).lower()

            if color in palette_colors:
                score += 0.8
            elif self._is_neutral(color):
                score += 0.4

            if color in rules.get("preferred_colors", []):
                score += 0.6

            if item_type in rules.get("avoided_items", []):
                score -= 2.0

        # =========================
        # 3. AESTHETIC BALANCE
        # =========================
        score += self._aesthetic_score(items)

        # =========================
        # 4. PERSONALIZATION (UPGRADED)
        # =========================
        personalization_weight = 0.3 + (confidence * 0.7)

        memory_score = self._memory_score(items, context)

        vector = self._build_outfit_embedding(items)

        # 🔥 SAFETY CHECK
        if not vector or len(vector) < 100:
            qdrant_score = 0.0
        else:
            qdrant_score = self._qdrant_score(context.get("user_id"), vector)

        score += memory_score * personalization_weight
        score += qdrant_score * personalization_weight

        # =========================
        # 5. EXPLORATION BOOST
        # =========================
        score += self._exploration_boost(items, exploration_factor)

        return round(score, 3)

    # =========================
    # 🔥 EXPLORATION
    # =========================
    def _exploration_boost(self, items: List[Dict[str, Any]], factor: float) -> float:

        if factor <= 0:
            return 0.0

        score = 0.0

        colors = [
            color_normalizer.normalize(i.get("color"))
            for i in items if i.get("color")
        ]

        if len(set(colors)) >= 3:
            score += 0.5 * factor

        styles = [
            str(i.get("style", "")).lower()
            for i in items if i.get("style")
        ]

        if len(set(styles)) >= 2:
            score += 0.4 * factor

        score += random.uniform(0, 0.3) * factor

        return score

    # =========================
    # MEMORY
    # =========================
    def _memory_score(self, items: List[Dict[str, Any]], context: Dict[str, Any]) -> float:

        memory = context.get("memory", {}).get("memory_signals", {})
        if not memory:
            return 0.0

        score = 0.0

        for item in items:
            color = color_normalizer.normalize(item.get("color"))
            style = str(item.get("style", "")).lower()
            item_type = str(item.get("type", "")).lower()

            if color in memory.get("liked_colors", []):
                score += 0.4

            if any(s in style for s in memory.get("preferred_styles", [])):
                score += 0.5

            if item_type in memory.get("disliked_items", []):
                score -= 1.2

        # 🔥 small stabilizer boost
        if memory.get("liked_colors"):
            score += 0.1

        return score

    # =========================
    # 🔥 EMBEDDING (UPGRADED)
    # =========================
    def _build_outfit_embedding(self, items: List[Dict[str, Any]]) -> List[float]:

        # TEXT
        text = " ".join([
            f"{i.get('type','')} {i.get('color','')} {i.get('style','')} {i.get('fabric','')}"
            for i in items
        ])

        text_vector = embedding_service.encode_text(text) or []

        # IMAGE
        image_vectors = []

        for item in items[:5]:  # performance guard
            image_url = item.get("image_url") or item.get("image")

            if image_url:
                vec = encode_image_url(image_url)
                if vec and len(vec) >= 100:
                    image_vectors.append(vec)

        image_vector = []
        if image_vectors:
            length = len(image_vectors[0])
            image_vector = [
                sum(vec[i] for vec in image_vectors) / len(image_vectors)
                for i in range(length)
            ]

        # COMBINE
        if text_vector and image_vector:
            return text_vector + image_vector

        return text_vector or image_vector or []

    # =========================
    # 🔥 QDRANT (UPGRADED)
    # =========================
    def _qdrant_score(self, user_id: str, vector: List[float]) -> float:

        if not user_id or not vector:
            return 0.0

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

            # 🔥 visual weighting
            score = (like_score * 1.3) - (dislike_score * 1.5)

            return max(-5.0, min(score, 5.0))

        except Exception:
            return 0.0

    # =========================
    # AESTHETIC
    # =========================
    def _aesthetic_score(self, items: List[Dict[str, Any]]) -> float:

        colors = [
            color_normalizer.normalize(i.get("color"))
            for i in items if i.get("color")
        ]

        unique_colors = len(set(colors))

        if unique_colors == 1:
            return 1.0
        elif unique_colors == 2:
            return 0.7
        elif unique_colors >= 3:
            return 0.5

        return 0.0

    # =========================
    # HELPERS
    # =========================
    def _is_neutral(self, color: str) -> bool:
        return color in ["black", "white", "grey", "gray", "beige", "navy", "cream"]


# Singleton
style_scorer = UnifiedStyleScorer()
