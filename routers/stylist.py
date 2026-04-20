import base64
import uuid
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from brain.outfit_pipeline import get_daily_outfits
from brain.personalization.style_dna_engine import style_dna_engine
from brain.engines.style_board_engine import style_board_engine
from brain.engines.style_board_renderer import style_board_renderer
from services import ai_gateway
from services.appwrite_proxy import AppwriteProxy

router = APIRouter()


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
    except Exception as exc:
        print(f"[item-suggestions] error={str(exc)}")
        return {
            "name": request.sub_category.title(),
            "tags": ["versatile", "casual"],
            "pairing_rules": [
                "Pair with neutral basics.",
                "Layer depending on weather.",
            ],
        }


class OutfitPipelineRequest(BaseModel):
    user_id: str
    query: str = "What should I wear today?"
    wardrobe: Any = None
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)


def _dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _visual_intelligence_from_outfit(outfit: Dict[str, Any]) -> Dict[str, Any]:
    parts = [
        _dict(outfit.get("top")),
        _dict(outfit.get("bottom")),
        _dict(outfit.get("dress")),
        _dict(outfit.get("shoes")),
    ] + [x for x in (outfit.get("accessories") or []) if isinstance(x, dict)]

    colors = [_safe_text(p.get("color")).lower() for p in parts if _safe_text(p.get("color"))]
    patterns = [_safe_text(p.get("pattern")).lower() for p in parts if _safe_text(p.get("pattern"))]
    styles = [_safe_text(p.get("style")).lower() for p in parts if _safe_text(p.get("style"))]
    return {
        "dominant_palette": sorted(set(colors))[:4],
        "pattern_mix": sorted(set(patterns))[:4],
        "style_signals": sorted(set(styles))[:4],
        "composition_score": float(outfit.get("score") or 0.0),
        "story": _safe_text(_dict(outfit.get("story")).get("subtitle") or outfit.get("explanation")),
    }


def _render_style_boards(cards: List[Dict[str, Any]], context: Dict[str, Any], style_dna: Dict[str, Any]) -> List[Dict[str, Any]]:
    rendered_boards: List[Dict[str, Any]] = []
    for idx, card in enumerate(cards):
        items = card.get("items", []) if isinstance(card, dict) else []
        if not items:
            continue

        board = style_board_engine.build_board({"items": items, "score": card.get("score")}, context)
        try:
            image_bytes = style_board_renderer.render(board)
            image_base64 = base64.b64encode(image_bytes).decode()
        except Exception as exc:
            print(f"[renderer] error={exc}")
            image_base64 = None

        rendered_boards.append(
            {
                "board_id": str(uuid.uuid4()),
                "type": "style",
                "label": card.get("title"),
                "score": card.get("score"),
                "aesthetic": board.get("aesthetic"),
                "vibe": board.get("vibe"),
                "items": items,
                "image_base64": image_base64,
                "board_payload": {
                    "items": items,
                    "aesthetic": board.get("aesthetic"),
                    "vibe": board.get("vibe"),
                    "score": card.get("score"),
                    "style_dna": style_dna,
                    "card_id": card.get("id") or f"outfit_card_{idx + 1}",
                },
            }
        )
    return rendered_boards


@router.post("/pipeline")
def run_outfit_pipeline(request: OutfitPipelineRequest):
    appwrite = AppwriteProxy()
    context = dict(request.context or {})
    context["query"] = request.query
    context["user_id"] = request.user_id

    wardrobe = request.wardrobe
    if wardrobe is None:
        try:
            wardrobe = appwrite.list_documents("outfits", user_id=request.user_id)
        except Exception:
            wardrobe = []

    style_dna = style_dna_engine.build(
        {
            "user_id": request.user_id,
            "user_profile": request.user_profile or {},
            "history": context.get("history", []),
            "wardrobe": wardrobe,
        }
    )
    context["style_dna"] = style_dna

    try:
        result = get_daily_outfits(
            {
                "user_id": request.user_id,
                "wardrobe": wardrobe,
                "context": context,
            }
        )
        outfits = result.get("outfits") if isinstance(result.get("outfits"), list) else []
        cards = result.get("cards") if isinstance(result.get("cards"), list) else []
        board_item_ids = result.get("board_item_ids") if isinstance(result.get("board_item_ids"), list) else []

        if not outfits:
            return {
                "success": False,
                "board": "style",
                "type": "cards",
                "message": result.get("context") or "No outfits generated",
                "cards": [],
                "board_ids": "",
                "data": {
                    "outfits": [],
                    "visual_intelligence": {},
                    "pipeline": _dict(result.get("pipeline")),
                    "rendered_boards": [],
                },
                "meta": {
                    "count": 0,
                    "query": request.query,
                    "analysis_source": "outfit_pipeline",
                },
            }

        rendered_boards = _render_style_boards(cards, context, style_dna)
        visual_intelligence = _visual_intelligence_from_outfit(_dict(outfits[0])) if outfits else {}

        return {
            "success": True,
            "message": result.get("context") or "Outfits generated successfully",
            "board": "style",
            "type": "cards",
            "cards": cards,
            "board_ids": ",".join([str(x).strip() for x in board_item_ids if str(x).strip()]),
            "data": {
                "outfits": outfits,
                "visual_intelligence": visual_intelligence,
                "pipeline": _dict(result.get("pipeline")),
                "rendered_boards": rendered_boards,
            },
            "meta": {
                "count": len(cards),
                "query": request.query,
                "analysis_source": "outfit_pipeline",
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}")
