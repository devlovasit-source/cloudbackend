import random
from typing import Any, Dict, List

from brain.engines.style_graph_engine import style_graph_engine


class OutfitEngine:
    """
    🔥 ELITE OUTFIT ENGINE

    Responsibilities:
    - Build outfits (deterministic)
    - Score compatibility
    - Generate aesthetic signals
    - Generate description (non-LLM)

    NO LLM HERE. Pure intelligence layer.
    """

    # =========================
    # MAIN ENTRY
    # =========================
    def generate(self, wardrobe: List[Dict[str, Any]], context: Dict[str, Any]):

        if not wardrobe:
            return {"outfits": []}

        tops = [i for i in wardrobe if i.get("category") in ["top", "tops"]]
        bottoms = [i for i in wardrobe if i.get("category") in ["bottom", "bottoms"]]
        shoes = [i for i in wardrobe if i.get("category") in ["shoes", "footwear"]]
        layers = [i for i in wardrobe if i.get("category") in ["outerwear"]]
        accessories = [i for i in wardrobe if i.get("category") in ["accessories", "bags", "jewelry"]]

        graph = style_graph_engine.build_graph({
            "tops": tops,
            "bottoms": bottoms,
            "shoes": shoes
        })

        outfits = []

        for _ in range(6):  # generate multiple candidates
            top = self._pick_anchor(tops)
            bottom = self._best_match(top, bottoms, graph)
            shoe = self._best_match(top, shoes, graph)
            layer = self._optional(layers)
            accessory = self._optional(accessories)

            items = [x for x in [top, bottom, shoe, layer, accessory] if x]

            score = self._score_outfit(items, graph)
            aesthetic = self._build_aesthetic(items, context)
            description = self._build_description(items, aesthetic, context)

            outfits.append({
                "items": items,
                "score": round(score, 3),
                "aesthetic": aesthetic,
                "description": description
            })

        outfits.sort(key=lambda x: x["score"], reverse=True)

        return {
            "outfits": outfits[:3]
        }

    # =========================
    # PICKERS
    # =========================
    def _pick_anchor(self, tops):
        return random.choice(tops) if tops else None

    def _best_match(self, anchor, candidates, graph):
        if not anchor or not candidates:
            return None

        scored = []
        for c in candidates:
            weight = style_graph_engine.pair_weight(graph, anchor.get("id"), c.get("id"))
            scored.append((c, weight))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0] if scored else None

    def _optional(self, items):
        return random.choice(items) if items and random.random() > 0.5 else None

    # =========================
    # SCORING
    # =========================
    def _score_outfit(self, items, graph):

        score = 0

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                score += style_graph_engine.pair_weight(
                    graph,
                    items[i].get("id"),
                    items[j].get("id")
                )

        # diversity boost
        categories = set([i.get("category") for i in items])
        score += len(categories) * 0.2

        return score

    # =========================
    # AESTHETIC INTELLIGENCE
    # =========================
    def _build_aesthetic(self, items, context):

        colors = [i.get("color", "").lower() for i in items if i.get("color")]
        styles = [i.get("style", "").lower() for i in items if i.get("style")]

        # ---- COLOR STORY ----
        if len(set(colors)) == 1:
            color_story = "monochrome"
        elif len(set(colors)) <= 2:
            color_story = "balanced"
        else:
            color_story = "contrast"

        # ---- VIBE ----
        if "formal" in styles:
            vibe = "polished"
        elif "streetwear" in styles:
            vibe = "street"
        else:
            vibe = "clean_minimal"

        # ---- SILHOUETTE ----
        silhouette = "balanced"
        if any("oversized" in str(i.get("fit", "")).lower() for i in items):
            silhouette = "relaxed_top"
        if any("slim" in str(i.get("fit", "")).lower() for i in items):
            silhouette = "structured"

        return {
            "vibe": vibe,
            "color_story": color_story,
            "silhouette": silhouette
        }

    # =========================
    # DESCRIPTION (NO LLM)
    # =========================
    def _build_description(self, items, aesthetic, context):

        vibe = aesthetic.get("vibe")
        color = aesthetic.get("color_story")

        occasion = context.get("occasion") or "day out"

        if vibe == "polished":
            line = "Clean, sharp, and put-together."
        elif vibe == "street":
            line = "Relaxed with an edge."
        else:
            line = "Effortless and refined."

        color_line = {
            "monochrome": "The single-tone palette makes it feel elevated.",
            "balanced": "The color balance keeps it easy and wearable.",
            "contrast": "The contrast adds visual interest."
        }.get(color, "")

        return f"{line} Perfect for {occasion}. {color_line}".strip()


# Singleton
outfit_engine = OutfitEngine()
