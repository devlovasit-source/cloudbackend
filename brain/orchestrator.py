import base64
from typing import Any, Dict, List

from brain.nlu.intent_router import nlu_router

from brain.engines.outfit_engine import outfit_engine
from brain.engines.style_scorer import style_scorer
from brain.engines.style_board_engine import style_board_engine
from brain.engines.style_board_renderer import style_board_renderer

from brain.tone.archetype_learning_engine import archetype_learning_engine
from brain.response.response_assembler import response_assembler
from brain.tone.tone_engine import tone_engine

from services.qdrant_service import qdrant_service
from services.embedding_service import embedding_service


class Orchestrator:
    """
    🔥 ELITE ORCHESTRATOR

    Flow:
    Intent → Context → Outfit Generation → Scoring → Ranking → Boards → Memory → Response
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

        # 🔥 MODES
        if slots.get("mode") == "feed":
            return self._build_feed(context)

        if slots.get("mode") == "explore":
            return self._explore(context)

        if slots.get("mode") == "similar":
            return self._similar(context)

        if slots.get("feedback"):
            return self._feedback(context)

        # 🔥 ROUTING
        if intent == "styling":
            result = self._handle_styling(context)
        else:
            result = {"message": "Tell me what you need — styling, meals, or plans."}

        # 🔥 RESPONSE ASSEMBLY
        final_response = response_assembler.assemble(result, context)

        # 🔥 FINAL TONE
        final_response["message"] = tone_engine.apply(
            final_response.get("message", ""),
            user_profile=context.get("user_profile"),
            signals={
                "context_mode": "styling",
                "aesthetic": context.get("aesthetic"),
            },
        )

        return final_response

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
            "slots": slots,
        }

    # =========================
    # 🔥 STYLING CORE (UPGRADED)
    # =========================
    def _handle_styling(self, context):

        result = outfit_engine.generate(context)
        outfits = result.get("outfits", [])

        if not outfits:
            return {"message": "I couldn't build outfits yet."}

        scored_outfits = []

        # 🔥 CORE INTELLIGENCE LAYER
        for o in outfits:

            # embedding
            o["embedding"] = self._embed_outfit(o, context)

            # 🔥 SCORING (MAIN BRAIN)
            score_data = style_scorer.score_outfit(o, context)

            o["final_score"] = score_data.get("score", 0)
            o["label"] = score_data.get("label", "Look")
            o["reasons"] = score_data.get("reasons", [])

            scored_outfits.append(o)

        # 🔥 SORT BY INTELLIGENCE
        scored_outfits.sort(key=lambda x: x["final_score"], reverse=True)

        selected = scored_outfits[:3]

        boards = []

        for outfit in selected:
            board = style_board_engine.build_board(outfit, context)
            image = style_board_renderer.render(board)

            embedding = outfit.get("embedding")

            # 🔥 MEMORY STORAGE (QDRANT)
            qdrant_service.upsert_style_board(
                board_id=outfit.get("id"),
                vector=embedding,
                payload={
                    "userId": context.get("user_id"),
                    "aesthetic": outfit.get("aesthetic"),
                    "occasion": context.get("occasion"),
                },
            )

            boards.append({
                "id": outfit.get("id"),
                "image_base64": base64.b64encode(image).decode(),
                "embedding": embedding,
                "aesthetic": outfit.get("aesthetic"),
                "description": outfit.get("description"),
                "score": outfit.get("final_score"),
                "label": outfit.get("label"),
                "reasons": outfit.get("reasons"),
            })

        context["aesthetic"] = selected[0].get("aesthetic")

        return {
            "type": "styling",
            "outfits": selected,
            "boards": boards,
            "message": selected[0].get("description", "")
        }

    # =========================
    # FEED
    # =========================
    def _build_feed(self, context):

        result = outfit_engine.generate(context)
        outfits = result.get("outfits", [])

        enriched = []

        for o in outfits:
            o["embedding"] = self._embed_outfit(o, context)

            score_data = style_scorer.score_outfit(o, context)
            o["final_score"] = score_data.get("score", 0)

            enriched.append(o)

        enriched.sort(key=lambda x: x["final_score"], reverse=True)

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

        return {"type": "feed", "data": feed}

    # =========================
    # EXPLORE
    # =========================
    def _explore(self, context):

        boards = qdrant_service.get_all_boards(limit=50)

        user_vector = self._build_user_vector(
            context.get("user_memory", {})
        )

        ranked = self._rank_boards(boards, user_vector)

        return {"type": "explore", "data": ranked[:20]}

    # =========================
    # SIMILAR
    # =========================
    def _similar(self, context):

        embedding = context.get("embedding")

        results = qdrant_service.search_similar_boards(
            embedding,
            limit=15
        )

        return {"type": "similar", "data": results}

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

        return {"type": "feedback", "data": memory}

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
