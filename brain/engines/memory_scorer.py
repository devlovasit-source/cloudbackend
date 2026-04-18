from typing import Dict, List, Any
import time
from services.qdrant_service import qdrant_service


class MemoryScorer:
    """
    🔥 ELITE MEMORY SCORER

    Combines:
    - temporal embedding memory
    - explicit signals
    - session awareness (highest priority)
    - collaborative filtering (similar users)

    Final range: ~ -4 to +4
    """

    HALF_LIFE_HOURS = 72  # temporal decay

    # =========================
    # MAIN API
    # =========================
    def score(self, embedding: List[float], context: Dict[str, Any]) -> float:

        if not embedding:
            return 0.0

        score = 0.0

        # -------------------------
        # PRIORITY ORDER
        # -------------------------
        score += self._session_score(context)                # 🔥 strongest
        score += self._embedding_score(context, embedding)   # temporal
        score += self._explicit_score(context)               # signals
        score += self._cluster_score(context, embedding)     # collaborative

        # -------------------------
        # STABILITY CLAMP
        # -------------------------
        return max(-4.0, min(score, 4.0))

    # =========================
    # 🔥 SESSION (HIGHEST PRIORITY)
    # =========================
    def _session_score(self, context: Dict[str, Any]) -> float:

        session = context.get("session", {})
        if not session:
            return 0.0

        score = 0.0

        # latest refinement dominates
        history = session.get("refinement_history", [])

        if history:
            latest = history[-1]["value"]

            if latest == "sharp":
                score += 0.8
            elif latest == "relaxed":
                score += 0.8
            elif latest == "minimal":
                score += 0.7
            elif latest == "bold":
                score += 0.7

        # occasion override
        if session.get("occasion"):
            score += 0.5

        return score

    # =========================
    # 🔥 TIME DECAY
    # =========================
    def _time_weight(self, timestamp: float) -> float:

        if not timestamp:
            return 0.5

        age_hours = (time.time() - timestamp) / 3600
        weight = 0.5 ** (age_hours / self.HALF_LIFE_HOURS)

        return max(0.1, min(weight, 1.0))

    # =========================
    # 🔥 EMBEDDING MEMORY
    # =========================
    def _embedding_score(self, context: Dict, embedding: List[float]) -> float:

        user_id = context.get("user_id")
        if not user_id:
            return 0.0

        try:
            liked = qdrant_service.search_user_memory(
                user_id=user_id,
                vector=embedding,
                memory_type="liked",
                limit=5
            )

            disliked = qdrant_service.search_user_memory(
                user_id=user_id,
                vector=embedding,
                memory_type="disliked",
                limit=5
            )

            score = 0.0

            # liked
            for r in liked:
                sim = r.get("score", 0)
                ts = r.get("timestamp")
                w = self._time_weight(ts)

                score += sim * w * 1.4

            # disliked
            for r in disliked:
                sim = r.get("score", 0)
                ts = r.get("timestamp")
                w = self._time_weight(ts)

                score -= sim * w * 1.7

            return score

        except Exception:
            return 0.0

    # =========================
    # 🔥 EXPLICIT SIGNALS
    # =========================
    def _explicit_score(self, context: Dict[str, Any]) -> float:

        memory = context.get("user_memory", {}) or {}
        signals = memory.get("memory_signals", {})

        if not signals:
            return 0.0

        score = 0.0

        ts = signals.get("timestamp")
        weight = self._time_weight(ts)

        if signals.get("preferred_styles"):
            score += 0.4 * weight

        if signals.get("liked_colors"):
            score += 0.3 * weight

        if signals.get("explore_colors"):
            score += 0.15 * weight

        return score

    # =========================
    # 🔥 COLLABORATIVE FILTERING
    # =========================
    def _cluster_score(self, context: Dict, embedding: List[float]) -> float:

        user_id = context.get("user_id")
        memory = context.get("user_memory", {})

        try:
            from brain.user_profile_vector import user_profile_vector
            from brain.user_cluster_engine import user_cluster_engine

            user_vec = user_profile_vector.build(memory)

            if not user_vec:
                return 0.0

            similar_users = user_cluster_engine.find_similar_users(user_id, user_vec)

            if not similar_users:
                return 0.0

            score = 0.0

            for u in similar_users:
                sim = u.get("score", 0)

                # weaker than personal memory
                score += sim * 0.4

            return score

        except Exception:
            return 0.0


# singleton
memory_scorer = MemoryScorer()
