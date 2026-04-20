from __future__ import annotations

from typing import Any, Dict, List

from brain.daily_dependency_engine import build_daily_dependency_response
from brain.intent_engine import detect_intent
from brain.outfit_pipeline import get_daily_outfits
from brain.plan_pack_flow import build_plan_pack_response
from brain.response.response_assembler import response_assembler
from brain.tone.tone_engine import tone_engine
from services.appwrite_proxy import AppwriteProxy


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _extract_occasion(text: str, slots: Dict[str, Any], context: Dict[str, Any]) -> str:
    explicit = _safe_text(slots.get("occasion") or context.get("occasion"))
    if explicit:
        return explicit.lower()

    lowered = (text or "").lower()
    if "wedding" in lowered:
        return "wedding"
    if "party" in lowered:
        return "party"
    if "office" in lowered or "work" in lowered:
        return "office"
    if "date" in lowered:
        return "date_night"
    if "travel" in lowered or "trip" in lowered:
        return "travel"
    return ""


def _normalize_weather_context(context: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(context)
    weather_data = _dict(out.get("weather_data"))
    if not weather_data and isinstance(out.get("user_profile"), dict):
        weather_data = _dict(_dict(out["user_profile"]).get("weather"))
    out["weather_data"] = weather_data
    if not out.get("weather") and weather_data.get("condition"):
        out["weather"] = _safe_text(weather_data.get("condition")).lower()
    return out


def _wardrobe_from_appwrite(user_id: str) -> List[Dict[str, Any]]:
    try:
        docs = AppwriteProxy().list_documents("outfits", user_id=user_id, limit=200)
    except Exception:
        return []
    if not isinstance(docs, list):
        return []
    rows: List[Dict[str, Any]] = []
    for d in docs:
        if not isinstance(d, dict):
            continue
        rows.append(
            {
                "id": d.get("$id") or d.get("id"),
                "name": d.get("name"),
                "category": d.get("category") or d.get("main_category"),
                "sub_category": d.get("sub_category") or d.get("subcategory"),
                "color": d.get("color") or d.get("color_code"),
                "pattern": d.get("pattern"),
                "occasion_tags": d.get("occasions") or [],
                "weather_tags": d.get("weather") or [],
                "style": d.get("style") or d.get("vibe"),
                "fabric": d.get("fabric"),
                "fit": d.get("fit"),
            }
        )
    return rows


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


class AhviOrchestrator:
    def run(self, *, text: str, user_id: str | None = None, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        query = _safe_text(text)
        uid = _safe_text(user_id) or _safe_text(_dict(context).get("user_id")) or "user_1"
        ctx = _normalize_weather_context(_dict(context))
        user_profile = _dict(ctx.get("user_profile"))

        intent_row = detect_intent(query, history=(ctx.get("history") or []))
        intent = _safe_text(intent_row.get("intent")).lower() or "general"
        slots = _dict(intent_row.get("slots"))
        occasion = _extract_occasion(query, slots, ctx)
        if occasion:
            ctx["occasion"] = occasion

        if intent == "daily_dependency":
            out = build_daily_dependency_response(user_id=uid, context=ctx)
            out["meta"] = {**_dict(out.get("meta")), "intent": intent, "confidence": float(intent_row.get("confidence", 0.0))}
            return out

        if intent == "plan_pack":
            out = build_plan_pack_response(query, ctx)
            out["success"] = True
            out["meta"] = {**_dict(out.get("meta")), "intent": intent, "confidence": float(intent_row.get("confidence", 0.0))}
            return out

        if intent == "wardrobe_query":
            docs = _wardrobe_from_appwrite(uid)
            return {
                "success": True,
                "message": f"You currently have {len(docs)} items in your wardrobe.",
                "board": "wardrobe",
                "type": "stats",
                "cards": [],
                "data": {"total_items": len(docs)},
                "meta": {"intent": intent, "confidence": float(intent_row.get("confidence", 0.0))},
            }

        if intent in {"daily_outfit", "occasion_outfit", "explore_styles"}:
            wardrobe = ctx.get("wardrobe")
            if not isinstance(wardrobe, (list, dict)) or not wardrobe:
                wardrobe = _wardrobe_from_appwrite(uid)

            outfit_result = get_daily_outfits(
                {
                    "user_id": uid,
                    "wardrobe": wardrobe,
                    "context": {
                        **ctx,
                        "query": query,
                        "occasion": occasion or _safe_text(ctx.get("occasion")),
                    },
                }
            )

            outfits = outfit_result.get("outfits") if isinstance(outfit_result.get("outfits"), list) else []
            visual_intel = _visual_intelligence_from_outfit(_dict(outfits[0])) if outfits else {}

            message = _safe_text(outfit_result.get("context") or outfit_result.get("message"))
            if outfits and not message:
                message = _safe_text(_dict(_dict(outfits[0]).get("story")).get("subtitle") or _dict(outfits[0]).get("explanation"))
            if not message:
                message = "Your visual intelligence result is ready."

            first_outfit = _dict(outfits[0]) if outfits else {}
            toned = tone_engine.apply(
                message,
                user_profile=user_profile,
                signals={"context_mode": "styling", **_dict(ctx.get("signals"))},
                context={
                    "aesthetic": first_outfit.get("aesthetic"),
                    "outfit_data": {"items": [x for x in first_outfit.values() if isinstance(x, dict)]},
                },
            )

            return {
                "success": True,
                "message": toned,
                "board": "style",
                "type": "cards",
                "cards": outfit_result.get("cards") if isinstance(outfit_result.get("cards"), list) else [],
                "board_ids": ",".join(outfit_result.get("board_item_ids") or []),
                "data": {
                    "outfits": outfits,
                    "visual_intelligence": visual_intel,
                    "pipeline": _dict(outfit_result.get("pipeline")),
                },
                "meta": {
                    "intent": intent,
                    "confidence": float(intent_row.get("confidence", 0.0)),
                    "visual_intelligence_enabled": True,
                },
            }

        merged = {
            "type": "general",
            "message": "Tell me what you need: outfit, planning, or organizing help.",
            "data": {},
        }
        fallback = response_assembler.assemble(merged_output=merged, context={"user_profile": user_profile, "signals": {"context_mode": "home"}})
        return {
            "success": True,
            "message": fallback,
            "board": "general",
            "type": "text",
            "cards": [],
            "data": {},
            "meta": {"intent": intent, "confidence": float(intent_row.get("confidence", 0.0))},
        }


ahvi_orchestrator = AhviOrchestrator()
orchestrator = ahvi_orchestrator
