from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List
import base64
import uuid

from services import ai_gateway
from brain.personalization.style_dna_engine import style_dna_engine
from services.appwrite_proxy import AppwriteProxy

# 🔥 NEW ENGINES
from brain.engines.outfit_engine import outfit_engine
from brain.engines.style_board_engine import style_board_engine
from brain.engines.style_board_renderer import style_board_renderer

router = APIRouter()


# =========================
# ITEM CONTEXT (UNCHANGED)
# =========================
class ItemContextRequest(BaseModel):
    main_category: str
    sub_category: str
    color_hex: str


@router.post("/item-suggestions")
def get_item_suggestions(request: ItemContextRequest):

    system_instruction = (
        "You are Ahvi's Fashion Knowledge Engine. The user just uploaded a new garment. "
        "Return JSON with: name, tags (4), pairing_rules (2). Output ONLY JSON."
    )

    user_prompt = (
        f"Item: {request.sub_category}\n"
        f"Category: {request.main_category}\n"
        f"Color Hex: {request.color_hex}"
    )

    try:
        messages = [{"role": "user", "content": user_prompt}]
        return ai_gateway.chat_json_object(
            messages,
            system_instruction=system_instruction,
            model="llama3.1",
        )

    except Exception as e:
        print(f"[item-suggestions] error={str(e)}")

        return {
            "name": request.sub_category.title(),
            "tags": ["versatile", "casual"],
            "pairing_rules": [
                "Pair with neutral basics.",
                "Layer depending on weather."
            ]
        }


# =========================
# REQUEST MODEL
# =========================
class OutfitPipelineRequest(BaseModel):
    user_id: str
    query: str = "What should I wear today?"
    wardrobe: Any = None
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)


# =========================
# 🔥 MAIN PIPELINE (UPGRADED)
# =========================
@router.post("/pipeline")
def run_outfit_pipeline(request: OutfitPipelineRequest):

    appwrite = AppwriteProxy()

    # -------------------------
    # CONTEXT BUILD
    # -------------------------
    context = dict(request.context or {})
    context["query"] = request.query
    context["user_id"] = request.user_id

    # -------------------------
    # LOAD WARDROBE (IF NOT PASSED)
    # -------------------------
    wardrobe = request.wardrobe

    if wardrobe is None:
        try:
            wardrobe = appwrite.list_documents(
                "outfits",
                user_id=request.user_id
            )
        except Exception:
            wardrobe = []

    # -------------------------
    # BUILD STYLE DNA
    # -------------------------
    style_dna = style_dna_engine.build(
        {
            "user_id": request.user_id,
            "user_profile": request.user_profile or {},
            "history": context.get("history", []),
            "wardrobe": wardrobe,
        }
    )

    context["style_dna"] = style_dna

    # -------------------------
    # 🔥 GENERATE OUTFITS
    # -------------------------
    try:
        result = outfit_engine.generate(wardrobe, context)
        routes = result.get("routes", [])

        if not routes:
            return {
                "success": False,
                "message": "No outfits generated",
                "data": []
            }

        boards_output = []

        # -------------------------
        # 🔥 BUILD BOARDS + RENDER
        # -------------------------
        for route in routes:

            outfit = route.get("outfit", {})
            items = outfit.get("items", [])

            if not items:
                continue

            board = style_board_engine.build_board(outfit, context)

            try:
                image_bytes = style_board_renderer.render(board)
                image_base64 = base64.b64encode(image_bytes).decode()
            except Exception as e:
                print(f"[renderer] error={e}")
                image_base64 = None

            board_id = str(uuid.uuid4())

            boards_output.append({
                "board_id": board_id,
                "type": route.get("type"),
                "label": route.get("label"),
                "score": outfit.get("score"),
                "aesthetic": board.get("aesthetic"),
                "vibe": board.get("vibe"),
                "items": items,
                "image_base64": image_base64,

                # 🔥 FEEDBACK-READY PAYLOAD
                "board_payload": {
                    "items": items,
                    "aesthetic": board.get("aesthetic"),
                    "vibe": board.get("vibe"),
                    "score": outfit.get("score"),
                    "style_dna": style_dna
                }
            })

        return {
            "success": True,
            "message": "Outfits generated successfully",
            "data": boards_output,
            "meta": {
                "count": len(boards_output),
                "query": request.query
            }
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed: {exc}"
        )
