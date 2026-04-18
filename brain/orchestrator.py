import base64
from typing import Any, Dict, List

from brain.nlu.intent_router import nlu_router

from brain.engines.outfit_engine import outfit_engine
from brain.engines.style_graph_engine import style_graph_engine
from brain.engines.styling.palette_engine import palette_engine
from brain.engines.style_board_engine import style_board_engine
from brain.engines.style_board_renderer import style_board_renderer

from brain.tone.archetype_learning_engine import archetype_learning_engine

from brain.response.response_assembler import response_assembler
from brain.tone.tone_engine import tone_engine

from services.qdrant_service import qdrant_service
from services.embedding_service import encode_metadata


class Orchestrator:
    """
    🔥 FINAL INTELLIGENT ORCHESTRATOR

    Flow:
    Intent → Context → Outfit → Boards → Embeddings → Ranking → Response
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

        # 🔥 ASSEMBLY
        response = response_assembler.assemble(result, context)

        # 🔥 FINAL TONE
        response["message"] = tone_engine.apply(
            response.get("message", ""),
            user_profile=context.get("user_profile"),
            signals={
                "context_mode": "styling",
                "aesthetic": context.get("aesthetic"),
            },
        )

        return response

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
    # STYLING CORE
    # =========================
    def _handle_styling(self, context):

        result = outfit_engine.generate(context)
        outfits = result.get("outfits", [])

        if not outfits:
            return {"message": "I couldn't build outfits yet."}

        # 🔥 EMBEDDING + RANKING
        for o in outfits:
            o["embedding"] = self._embed_outfit(o, context)

        outfits = self._rank_outfits(outfits, context)

        selected = outfits[:3]

        boards = []

        for outfit in selected:
            board = style_board_engine.build_board(outfit, context)
            image = style_board_renderer.render(board)

            embedding = outfit.get("embedding")

            # 🔥 STORE IN QDRANT
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
            })

        context["aesthetic"] = selected[0].get("aesthetic")

        return {
            "type": "styling",
            "data": {
                "outfits": selected,
                "style_boards": boards
            },
            "message": selected[0].get("description", "")
        }

    # =========================
    # FEED (REELS)
    # =========================
    def _build_feed(self, context):

        result = outfit_engine.generate(context)
        outfits = result.get("outfits", [])

        for o in outfits:
            o["embedding"] = self._embed_outfit(o, context)

        outfits = self._rank_outfits(outfits, context)

        feed = []

        for o in outfits[:10]:
            board = style_board_engine.build_board(o, context)
            image = style_board_renderer.render(board)

            feed.append({
                "id": o.get("id"),
                "image_base64": base64.b64encode(image).decode(),
                "embedding": o.get("embedding"),
                "description": o.get("description"),
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

        return encode_metadata({
            "aesthetic": outfit.get("aesthetic"),
            "colors": outfit.get("colors"),
            "category": outfit.get("category"),
            "occasion": context.get("occasion"),
        })

    # =========================
    # RANKING
    # =========================
    def _rank_outfits(self, outfits, context):

        memory = context.get("user_memory", {})
        user_vector = self._build_user_vector(memory)

        def score(o):
            base = o.get("final_score", 0)

            if user_vector and o.get("embedding"):
                sim = qdrant_service.cosine_similarity(
                    user_vector, o["embedding"]
                )
                base += sim * 2

            return base

        return sorted(outfits, key=score, reverse=True)

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
