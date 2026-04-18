import base64
import time
from typing import Any, Dict

from brain.intent_engine import detect_intent

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

    def handle(self, user_input: str, user: Dict[str, Any]) -> Dict[str, Any]:

        context = self._build_context(user)

        print("\n=== PIPELINE START ===")
        print("INPUT:", user_input)

        intent_data = detect_intent(
            user_text=user_input,
            history=user.get("history", []),
            context=context
        )

        intent = intent_data.get("intent", "general")
        slots = intent_data.get("slots", {})

        print("INTENT:", intent_data)

        context.update({
            "intent": intent,
            "intent_meta": intent_data,
            "slots": slots,
            "user_input": user_input,
            "occasion": slots.get("occasion"),
            "weather": slots.get("weather"),
            "mode": slots.get("mode"),
        })

        # ✅ TEMP WARDROBE (ONLY IF EMPTY)
        if not context.get("wardrobe"):
            print("⚠️ Using TEMP wardrobe")
            context["wardrobe"] = self._get_temp_wardrobe()

        print("WARDROBE SIZE:", len(context.get("wardrobe", [])))

        context["signals"] = {
            "intent": intent,
            "intent_source": intent_data.get("source"),
            "confidence": intent_data.get("confidence"),
            "occasion": context.get("occasion"),
            "weather": context.get("weather"),
            "user_memory": context.get("user_memory"),
            "session": context.get("session"),
        }

        context = proactive_engine.inject(context)

        # -------------------------
        # ROUTING (UNCHANGED)
        # -------------------------
        if context.get("proactive_signals"):
            result = self._handle_styling(context)

        elif slots.get("mode") == "feed":
            result = self._build_feed(context)

        elif slots.get("mode") == "explore":
            result = self._explore(context)

        elif slots.get("mode") == "similar":
            result = self._similar(context)

        elif slots.get("feedback"):
            result = self._feedback(context)

        elif intent == "styling":
            result = self._handle_styling(context)

        else:
            result = {
                "type": "text",
                "message": "Tell me what you need — styling, meals, or plans.",
                "data": {}
            }

        print("RESULT TYPE:", result.get("type"))

        context["signals"].update({
            "aesthetic": context.get("aesthetic"),
            "item_explanations": context.get("item_explanations"),
            "reasons": context.get("reasons"),
        })

        response = self._safe(
            lambda: response_assembler.assemble(result, context),
            {"message": {"role": "assistant", "content": "Something went wrong"}}
        )

        print("RESPONSE READY")
        print("=== PIPELINE END ===\n")

        return {
            "success": True,
            **response,
            "meta": {
                "type": result.get("type", "text"),
                "intent": intent,
                "source": intent_data.get("source"),
                "confidence": intent_data.get("confidence")
            }
        }

    # =========================
    # CONTEXT BUILDER
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
    # ✅ FIXED TEMP WARDROBE
    # =========================
    def _get_temp_wardrobe(self):
        return [
            {"type": "shirt", "color": "white", "category": "top", "style": "minimal"},
            {"type": "t-shirt", "color": "black", "category": "top", "style": "streetwear"},
            {"type": "jeans", "color": "blue", "category": "bottom", "style": "casual"},
            {"type": "trousers", "color": "beige", "category": "bottom", "style": "formal"},
            {"type": "sneakers", "color": "white", "category": "footwear", "style": "casual"},
            {"type": "loafers", "color": "brown", "category": "footwear", "style": "formal"},
        ]

    # =========================
    # SAFE EXECUTION
    # =========================
    def _safe(self, fn, fallback):
        try:
            return fn()
        except Exception as e:
            print("ERROR:", e)
            return fallback

    # =========================
    # STYLING CORE (UNCHANGED)
    # =========================
    def _handle_styling(self, context):

        result = self._safe(
            lambda: outfit_engine.generate(context),
            {"outfits": []}
        )

        outfits = result.get("outfits", [])

        print("OUTFITS GENERATED:", len(outfits))

        if not outfits:
            return {"type": "styling", "data": {}}

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

            print("SCORE:", final)

            scored.append(o)

        scored.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        selected = scored[:3]

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

            # 🔥 KEEP QDRANT (NOT REMOVED)
            self._safe(
                lambda: qdrant_service.upsert_style_board(
                    board_id=o.get("id"),
                    vector=o.get("embedding"),
                    payload={
                        "userId": context.get("user_id"),
                        "aesthetic": o.get("aesthetic"),
                        "occasion": context.get("occasion"),
                    },
                ),
                None
            )

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
            "data": {
                "outfits": selected,
                "boards": boards
            }
        }

    # =========================
    # KEEP ALL OTHER METHODS
    # =========================

    def _build_feed(self, context):
        return {
            "type": "feed",
            "data": []
        }

    def _explore(self, context):
        return {
            "type": "explore",
            "data": qdrant_service.get_all_boards(limit=50)
        }

    def _similar(self, context):
        return {
            "type": "similar",
            "data": qdrant_service.search_similar_boards(
                context.get("embedding"),
                limit=15
            )
        }

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

        self._safe(
            lambda: qdrant_service.upsert_user_profile(
                user_id=context.get("user_id"),
                vector=embedding,
                payload={"memory": memory}
            ),
            None
        )

        return {"type": "feedback", "data": memory}

    def _embed_outfit(self, outfit, context):

        return embedding_service.encode_metadata({
            "aesthetic": outfit.get("aesthetic"),
            "colors": outfit.get("colors"),
            "category": outfit.get("category"),
            "occasion": context.get("occasion"),
        })


ahvi_orchestrator = Orchestrator()
