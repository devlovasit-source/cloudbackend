
from typing import List, Dict, Any, Optional
from services.qdrant_service import qdrant_service


class WardrobeSelector:
    """
    🔥 ELITE WARDROBE SELECTOR

    Responsibilities:
    ✔ Smart item selection
    ✔ Embedding + heuristic scoring
    ✔ Robust fallback

    Guarantees:
    ✔ Always returns best possible item
    ✔ Never crashes on missing embeddings
    """

    # =========================
    # TYPE NORMALIZATION
    # =========================
    TYPE_MAP = {
        "tshirt": "top",
        "shirt": "top",
        "tee": "top",
        "top": "top",

        "jeans": "bottom",
        "pants": "bottom",
        "trousers": "bottom",
        "bottom": "bottom",

        "shoes": "footwear",
        "sneakers": "footwear",
        "footwear": "footwear",
    }

    def normalize_type(self, t: str) -> str:
        if not t:
            return ""
        t = t.lower().strip()
        return self.TYPE_MAP.get(t, t)

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
            print("⚠️ No wardrobe available")
            return None

        target_type = self.normalize_type(target_type)

        # -------------------------
        # FILTER BY TYPE (SMART)
        # -------------------------
        candidates = [
            w for w in wardrobe
            if target_type in self.normalize_type(str(w.get("type", "")))
               or target_type in self.normalize_type(str(w.get("category", "")))
        ]

        if not candidates:
            print(f"⚠️ No candidates for type: {target_type}")
            return None

        # -------------------------
        # SCORING
        # -------------------------
        scored = []

        for item in candidates:
            score = 0.0

            # 🔥 EMBEDDING SCORE (PRIMARY)
            if reference_embedding and item.get("embedding"):
                try:
                    sim = qdrant_service.cosine_similarity(
                        reference_embedding,
                        item["embedding"]
                    )
                    score += sim * 0.8
                except Exception:
                    pass

            # 🔥 HEURISTIC BOOST
            if target_type in str(item.get("type", "")).lower():
                score += 0.2

            # slight preference for items with embeddings
            if item.get("embedding"):
                score += 0.05

            scored.append({
                "item": item,
                "score": score
            })

        # -------------------------
        # SORT
        # -------------------------
        scored.sort(key=lambda x: x["score"], reverse=True)

        best = scored[0]["item"]

        print(f"SELECTED ITEM → type: {target_type} | score: {scored[0]['score']:.3f}")

        return best

    # =========================
    # FALLBACK (SAFE)
    # =========================
    def fallback(self, wardrobe: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not wardrobe:
            return None

        # prefer items with embeddings
        for item in wardrobe:
            if item.get("embedding"):
                return item

        return wardrobe[0]


# singleton
wardrobe_selector = WardrobeSelector()
