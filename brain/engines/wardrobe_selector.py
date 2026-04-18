from typing import List, Dict, Any, Optional
from services.qdrant_service import qdrant_service


class WardrobeSelector:
    """
    🔥 ADVANCED WARDROBE SELECTOR

    Chooses best real item from user's wardrobe using:
    - embedding similarity
    - memory preferences
    - style DNA alignment
    - color harmony
    """

    # =========================
    # MAIN API
    # =========================
    def find_best_match(
        self,
        target_type: str,
        context: Dict[str, Any],
        reference_embedding: Optional[List[float]] = None,
    ) -> Optional[Dict]:

        wardrobe = context.get("wardrobe", [])
        if not wardrobe:
            return None

        candidates = [
            w for w in wardrobe
            if target_type in str(w.get("type", "")).lower()
        ]

        if not candidates:
            return None

        scored = []

        for item in candidates:
            score = self._score_item(item, context, reference_embedding)
            scored.append((item, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[0][0] if scored else None

    # =========================
    # SCORING ENGINE
    # =========================
    def _score_item(self, item, context, reference_embedding):

        score = 0.0

        # -------------------------
        # 1. EMBEDDING SIMILARITY
        # -------------------------
        emb = item.get("embedding")

        if reference_embedding and emb:
            try:
                sim = qdrant_service.cosine_similarity(reference_embedding, emb)
                score += sim * 2.0  # strong signal
            except Exception:
                pass

        # -------------------------
        # 2. MEMORY ALIGNMENT
        # -------------------------
        memory = context.get("user_memory", {}).get("memory_signals", {})

        if emb:
            score += self._memory_score(emb, memory)

        # -------------------------
        # 3. STYLE DNA ALIGNMENT
        # -------------------------
        dna = context.get("style_dna", {})

        score += self._dna_score(item, dna)

        # -------------------------
        # 4. COLOR HARMONY
        # -------------------------
        score += self._color_score(item, context)

        return score

    # =========================
    # MEMORY SCORING
    # =========================
    def _memory_score(self, emb, memory):

        if not memory:
            return 0

        score = 0

        try:
            liked = memory.get("liked_embeddings", [])
            disliked = memory.get("disliked_embeddings", [])

            if liked:
                sim = max(
                    [qdrant_service.cosine_similarity(emb, l) for l in liked if l],
                    default=0
                )
                score += sim * 1.5

            if disliked:
                sim = max(
                    [qdrant_service.cosine_similarity(emb, d) for d in disliked if d],
                    default=0
                )
                score -= sim * 2.0

        except Exception:
            pass

        return score

    # =========================
    # STYLE DNA SCORING
    # =========================
    def _dna_score(self, item, dna):

        if not dna:
            return 0

        score = 0

        preferred_styles = dna.get("preferred_styles", [])
        preferred_colors = dna.get("preferred_colors", [])

        style = str(item.get("style", "")).lower()
        color = str(item.get("color", "")).lower()

        if style in preferred_styles:
            score += 0.8

        if color in preferred_colors:
            score += 0.6

        return score

    # =========================
    # COLOR LOGIC
    # =========================
    def _color_score(self, item, context):

        palette = context.get("palette", [])
        color = str(item.get("color", "")).lower()

        if not palette:
            return 0

        if color in palette:
            return 0.7

        if self._is_neutral(color):
            return 0.4

        return 0

    def _is_neutral(self, color: str) -> bool:
        return color in [
            "black", "white", "grey", "gray",
            "beige", "cream", "navy"
        ]


# singleton
wardrobe_selector = WardrobeSelector()
