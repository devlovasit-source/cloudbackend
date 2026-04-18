
import base64
import time
from typing import Any, Dict, List

from brain.intent_engine import detect_intent
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
    🔥 ELITE ORCHESTRATOR (FINAL FORM)

    Flow:
    Input → Hybrid Intent → Context → Proactive → Outfit → Refinement
          → Scoring → Boards → Signals → Response → Memory Update
    """

    # =========================
    # MAIN ENTRY
    # =========================
    def handle(self, user_input: str, user: Dict[str, Any]) -> Dict[str, Any]:

        context = self._build_context(user)

        # -------------------------
        # 🔥 HYBRID INTENT
        # -------------------------
        intent_data = detect_intent(
            user_text=user_input,
            history=user.get("history", []),
            context=context
        )

        intent = intent_data.get("intent")
        slots = intent_data.get("slots", {})

        context.update({
            "intent": intent,
            "intent_meta": intent_data,
            "slots": slots,
            "user_input": user_input
        })

        # -------------------------
        # 🔥 PROACTIVE FIRST
        # -------------------------
        context = proactive_engine.inject(context)

        # -------------------------
        # MODE ROUTING
        # -------------------------
        if slots.get("mode") == "feed":
            result = self._build_feed(context)

        elif slots.get("mode") == "explore":
            result = self._explore(context)

        elif slots.get("mode") == "similar":
            result = self._similar(context)

        elif slots.get("feedback"):
            result = self._feedback(context)

        elif context.get("proactive_signals"):
            result = self._handle_styling(context)

        elif intent == "styling":
            result = self._handle_styling(context)

        else:
            result = {
                "type": "text",
                "message": "Tell me what you need — styling, meals, or plans.",
                "data": {}
            }

        # -------------------------
        # 🔥 SIGNAL LAYER (CRITICAL)
        # -------------------------
        context["signals"] = {
            "intent": intent,
            "intent_source": intent_data.get("source"),
            "occasion": context.get("occasion"),
            "weather": context.get("weather"),
            "conversation_profile": context.get("conversation_profile"),
            "user_memory": context.get("user_memory"),
            "aesthetic": context.get("aesthetic"),
            "item_explanations": context.get("item_explanations"),
            "reasons": context.get("reasons"),
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
                "type": result.get("type", "text"),
                "intent": intent,
                "source": intent_data.get("source")
            }
        }

    # =========================
    # CONTEXT
    # =========================
    def _build_context(self, user):

        memory = user.get("memory", {}) or {}

        return {
            "user_id": user.get("user_id"),
            "wardrobe": user.get("wardrobe", []),
            "style_dna": user.get("style_dna", {}),
            "user_profile": user.get("profile", {}),
            "user_memory": memory,
            "conversation_profile": memory.get("conversation_memory", {}),
            "session": user.get("session", {}),
            "refinement": user.get("refinement"),
        }

    # =========================
    # 🔥 STYLING CORE
    # =========================
    def _handle_styling(self, context):

        result = outfit_engine.generate(context)
        outfits = result.get("outfits", [])

        if not outfits:
            return {"type": "styling", "message": "No outfits yet.", "data": {}}

        outfits = refinement_engine.apply(outfits, context)

        scored = []

        for o in outfits:

            o["embedding"] = self._embed_outfit(o, context)

            base = style_scorer.score_outfit(o, context)
            base_score = base.get("score", 0)

            memory_score = memory_scorer.score(
                o.get("embedding"),
                context.get("user_memory", {})
            )

            final = base_score + memory_score

            o.update({
                "final_score": final,
                "memory_score": memory_score,
                "label": base.get("label"),
                "reasons": base.get("reasons", [])
            })

            scored.append(o)

        scored.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        selected = scored[:3]

        # 🔥 EXPLAINABILITY PIPELINE
        context["item_explanations"] = [
            o.get("item_explanations") for o in selected
        ]
        context["reasons"] = [
            o.get("reasons") for o in selected
        ]

        boards = []

        for o in selected:

            board = style_board_engine.build_board(o, context)
            image = style_board_renderer.render(board)

            try:
                qdrant_service.upsert_style_board(
                    board_id=o.get("id"),
                    vector=o.get("embedding"),
                    payload={
                        "userId": context.get("user_id"),
                        "aesthetic": o.get("aesthetic"),
                        "occasion": context.get("occasion"),
                    },
                )
            except Exception:
                pass

            boards.append({
                "id": o.get("id"),
                "image_base64": base64.b64encode(image).decode(),
                "embedding": o.get("embedding"),
                "aesthetic": o.get("aesthetic"),
                "description": o.get("description"),
                "score": o.get("final_score"),
                "label": o.get("label"),
                "reasons": o.get("reasons"),
                "refined": o.get("refined"),
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
    # FEED / EXPLORE / SIMILAR
    # =========================
    def _build_feed(self, context):
        result = outfit_engine.generate(context)
        outfits = refinement_engine.apply(result.get("outfits", []), context)

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

        return {
            "type": "feed",
            "data": enriched[:10]
        }

    def _explore(self, context):
        boards = qdrant_service.get_all_boards(limit=50)
        return {"type": "explore", "data": boards}

    def _similar(self, context):
        return {
            "type": "similar",
            "data": qdrant_service.search_similar_boards(
                context.get("embedding"),
                limit=15
            )
        }

    # =========================
    # FEEDBACK (LEARNING)
    # =========================
    def _feedback(self, context):

        memory = context.get("user_memory", {})
        embedding = context.get("embedding")
        signals = context.get("slots", {})

        if signals.get("feedback") == "like" and embedding:
            memory.setdefault("liked_embeddings", []).append({
                "value": embedding,
                "ts": time.time()
            })

        if signals.get("feedback") == "dislike" and embedding:
            memory.setdefault("disliked_embeddings", []).append({
                "value": embedding,
                "ts": time.time()
            })

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


# singleton
ahvi_orchestrator = Orchestrator()
