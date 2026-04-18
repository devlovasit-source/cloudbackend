import base64
from typing import Any, Dict, List

from brain.nlu.intent_router import nlu_router

from brain.engines.outfit_engine import outfit_engine
from brain.engines.style_scorer import style_scorer
from brain.engines.style_board_engine import style_board_engine
from brain.engines.style_board_renderer import style_board_renderer
from brain.engines.refinement_engine import refinement_engine
from brain.engines.memory_scorer import memory_scorer
from brain.engines.proactive_engine import proactive_engine

from brain.tone.archetype_learning_engine import archetype_learning_engine
from brain.response.response_assembler import response_assembler

from services.qdrant_service import qdrant_service
from services.embedding_service import embedding_service


class Orchestrator:
    """
    🔥 ELITE ORCHESTRATOR

    Flow:
    Intent → Context → Proactive → Outfit → Refinement → Scoring → Boards → Response
    """

    # =========================
    # MAIN ENTRY
    # =========================
    def handle(self, user_input: str, user: Dict[str, Any]) -> Dict[str, Any]:

        intent_data = nlu_router.classify_intent(user_input)
        intent = intent_data.get("intent")
        slots = intent_data.get("slots", {})

        context = self._build_context(user, slots)
        context["intent"] = intent
        context["user_input"] = user_input

        # 🔥 PROACTIVE INTELLIGENCE
        context = proactive_engine.inject(context)

        # -------------------------
        # MODES
        # -------------------------
        if slots.get("mode") == "feed":
            return self._build_feed(context)

        if slots.get("mode") == "explore":
            return self._explore(context)

        if slots.get("mode") == "similar":
            return self._similar(context)

        if slots.get("feedback"):
            return self._feedback(context)

        # -------------------------
        # MAIN ROUTING
        # -------------------------
        if intent == "styling" or context.get("proactive_signals"):
            result = self._handle_styling(context)
        else:
            result = {
                "type": "text",
                "message": "Tell me what you need — styling, meals, or plans.",
                "data": {}
            }

        # -------------------------
        # RESPONSE
        # -------------------------
        message = response_assembler.assemble(result, context)

        return {
            "success": True,
            "message": message,
            "data": result.get("data", {}),
            "meta": {
                "type": result.get("type", "text")
            }
        }

    # =========================
    # CONTEXT
    # =========================
    def _build_context(self, user, slots):

        return {
            "user_id": user.get("user_id"),
            "wardrobe": user.get("wardrobe", []),
            "style_dna": user.get("style_dna", {}),
            "user_profile": user.get("profile", {}),
            "user_memory": user.get("memory", {}),
            "occasion": slots.get("occasion"),
            "weather": slots.get("weather"),
            "refinement": user.get("refinement"),
            "slots": slots,
        }

    # =========================
    # 🔥 STYLING CORE (ELITE)
    # =========================
    def _handle_styling(self, context):

        result = outfit_engine.generate(context)
        outfits = result.get("outfits", [])

        if not outfits:
            return {
                "type": "styling",
                "message": "I couldn't build outfits yet.",
                "data": {}
            }

        # 🔥 APPLY REFINEMENT (REAL SWAPS)
        outfits = refinement_engine.apply(outfits, context)

        scored_outfits = []

        for o in outfits:

            # -------------------------
            # EMBEDDING
            # -------------------------
            o["embedding"] = self._embed_outfit(o, context)

            # -------------------------
            # BASE SCORING
            # -------------------------
            score_data = style_scorer.score_outfit(o, context)
            base_score = score_data.get("score", 0)

            # -------------------------
            # 🔥 MEMORY PERSONALIZATION
            # -------------------------
            memory_score = memory_scorer.score(
                o.get("embedding"),
                context.get("user_memory", {})
            )

            final_score = base_score + memory_score

            o["final_score"] = final_score
            o["memory_score"] = memory_score
            o["label"] = score_data.get("label", "Look")
            o["reasons"] = score_data.get("reasons", [])

            scored_outfits.append(o)

        # -------------------------
        # SORT
        # -------------------------
        scored_outfits.sort(key=lambda x: x.get("final_score", 0), reverse=True)

        selected = scored_outfits[:3]

        boards = []

        for outfit in selected:

            board = style_board_engine.build_board(outfit, context)
            image = style_board_renderer.render(board)

            embedding = outfit.get("embedding")

            # 🔥 SAFE QDRANT UPSERT
            try:
                qdrant_service.upsert_style_board(
                    board_id=outfit.get("id"),
                    vector=embedding,
                    payload={
                        "userId": context.get("user_id"),
                        "aesthetic": outfit.get("aesthetic"),
                        "occasion": context.get("occasion"),
                    },
                )
            except Exception:
                pass

            boards.append({
                "id": outfit.get("id"),
                "image_base64": base64.b64encode(image).decode(),
                "embedding": embedding,
                "aesthetic": outfit.get("aesthetic"),
                "description": outfit.get("description"),
                "score": outfit.get("final_score"),
                "label": outfit.get("label"),
                "reasons": outfit.get("reasons"),
                "refined": outfit.get("refined"),
            })

        context["aesthetic"] = selected[0].get("aesthetic")

        return {
            "type": "styling",
            "message": selected[0].get("description", ""),
            "data": {
                "outfits": selected,
                "boards": boards
            }
        }

    # =========================
    # FEED
    # =========================
    def _build_feed(self, context):

        result = outfit_engine.generate(context)
        outfits = result.get("outfits", [])

        outfits = refinement_engine.apply(outfits, context)

        enriched = []

        for o in outfits:
            o["embedding"] = self._embed_outfit(o, context)

            base = style_scorer.score_outfit(o, context).get("score", 0)

            memory_score = memory_scorer.score(
                o.get("embedding"),
                context.get("user_memory", {})
            )

            o["final_score"] = base + memory_score
            enriched.append(o)

        enriched.sort(key=lambda x: x.get("final_score", 0), reverse=True)

        feed = []

        for o in enriched[:10]:
            board = style_board_engine.build_board(o, context)
            image = style_board_renderer.render(board)

            feed.append({
                "id": o.get("id"),
                "image_base64": base64.b64encode(image).decode(),
                "embedding": o.get("embedding"),
                "description": o.get("description"),
                "score": o.get("final_score"),
            })

        return {
            "type": "feed",
            "data": feed
        }

    # =========================
    # EXPLORE
    # =========================
    def _explore(self, context):

        boards = qdrant_service.get_all_boards(limit=50)

        user_vector = self._build_user_vector(
            context.get("user_memory", {})
        )

        ranked = self._rank_boards(boards, user_vector)

        return {
            "type": "explore",
            "data": ranked[:20]
        }

    # =========================
    # SIMILAR
    # =========================
    def _similar(self, context):

        embedding = context.get("embedding")

        results = qdrant_service.search_similar_boards(
            embedding,
            limit=15
        )

        return {
            "type": "similar",
            "data": results
        }

    # =========================
    # FEEDBACK
    # =========================
    def _feedback(self, context):

        signals = context.get("slots", {})
        memory = context.get("user_memory", {})

        embedding = context.get("embedding")

        if signals.get("feedback") == "like" and embedding:
            memory.setdefault("liked_embeddings", []).append(embedding)

        if signals.get("feedback") == "dislike" and embedding:
            memory.setdefault("disliked_embeddings", []).append(embedding)

        memory = archetype_learning_engine.update(memory, signals)

        return {
            "type": "feedback",
            "data": memory
        }

    # =========================
    # EMBEDDING
    # =========================
    def _embed_outfit(self, outfit, context):

        return embedding_service.encode_metadata({
            "aesthetic": outfit.get("aesthetic"),
            "colors": outfit.get("colors"),
            "category": outfit.get("category"),
            "occasion": context.get("occasion"),
        })

    # =========================
    # RANKING HELPERS
    # =========================
    def _rank_boards(self, boards, user_vector):

        if not user_vector:
            return boards

        def score(b):
            emb = b.get("embedding")
            if not emb:
                return 0
            return qdrant_service.cosine_similarity(user_vector, emb)

        return sorted(boards, key=score, reverse=True)

    def _build_user_vector(self, memory):

        liked = memory.get("liked_embeddings", [])

        if not liked:
            return None

        return [sum(x)/len(x) for x in zip(*liked)]


# Singleton
ahvi_orchestrator = Orchestrator()
