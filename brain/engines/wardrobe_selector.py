from typing import List, Dict, Any, Optional
from services.qdrant_service import qdrant_service


class WardrobeSelector:
    """
    🔥 CLEAN WARDROBE SELECTOR (FINAL)

    Responsibility:
    - Select best matching item from user's wardrobe

    DOES:
    ✔ filter by type
    ✔ embedding similarity (primary signal)
    ✔ safe fallback

    DOES NOT:
    ❌ memory scoring
    ❌ style DNA scoring
    ❌ palette scoring
    ❌ refinement scoring
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

        target_type = str(target_type or "").lower().strip()

        # -------------------------
        # FILTER BY TYPE
        # -------------------------
        candidates = [
            w for w in wardrobe
            if target_type in str(w.get("type", "")).lower()
        ]

        if not candidates:
            return None

        # -------------------------
        # EMBEDDING MATCH (PRIMARY)
        # -------------------------
        if reference_embedding:
            best = self._best_by_embedding(candidates, reference_embedding)
            if best:
                return best

        # -------------------------
        # FALLBACK STRATEGY
        # -------------------------
        return self._fallback(candidates)

    # =========================
    # EMBEDDING MATCH
    # =========================
    def _best_by_embedding(
        self,
        candidates: List[Dict[str, Any]],
        reference_embedding: List[float],
    ) -> Optional[Dict[str, Any]]:

        best_item = None
        best_score = float("-inf")

        for item in candidates:
            emb = item.get("embedding")
            if not emb:
                continue

            try:
                sim = qdrant_service.cosine_similarity(reference_embedding, emb)

                if sim > best_score:
                    best_score = sim
                    best_item = item

            except Exception:
                continue

        return best_item

    # =========================
    # FALLBACK
    # =========================
    def _fallback(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Safe fallback when:
        - no embeddings available
        - similarity fails
        """

        # prefer items with embeddings
        for item in candidates:
            if item.get("embedding"):
                return item

        # otherwise return first item
        return candidates[0]


# singleton
wardrobe_selector = WardrobeSelector()
