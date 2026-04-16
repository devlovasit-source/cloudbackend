from typing import Any, Dict, List
import random
import itertools

from brain.engines.style_graph_engine import style_graph_engine
from brain.engines.style_rules_engine import style_engine
from brain.engines.palette_engine import palette_engine
from brain.engines.unified_style_scorer import style_scorer


class OutfitEngine:
    """
    🔥 ELITE V4 — TRUE STYLIST ENGINE

    - Multi-combination generation
    - UnifiedStyleScorer driven ranking
    - Multi-aesthetic aware
    - Route-based output (safe / elevated / bold)
    """

    # =========================
    # MAIN ENTRY
    # =========================
    def generate(self, wardrobe: List[Dict[str, Any]], context: Dict[str, Any]):

        if not wardrobe:
            return {"routes": []}

        # ---- SPLIT ----
        tops = [i for i in wardrobe if i.get("category") in ["top", "tops"]]
        bottoms = [i for i in wardrobe if i.get("category") in ["bottom", "bottoms"]]
        shoes = [i for i in wardrobe if i.get("category") in ["shoes", "footwear"]]
        layers = [i for i in wardrobe if i.get("category") in ["outerwear"]]
        accessories = [i for i in wardrobe if i.get("category") in ["accessories", "bags", "jewelry"]]

        # ---- GRAPH ----
        graph = style_graph_engine.build_graph({
            "tops": tops,
            "bottoms": bottoms,
            "shoes": shoes
        })

        # ---- GENERATE COMBINATIONS 🔥 ----
        outfits = self._generate_candidates(tops, bottoms, shoes, layers, accessories)

        # ---- SCORE USING ELITE SCORER 🔥 ----
        scored = []

        for items in outfits:
            score = style_scorer.score_outfit(items, context, graph)
            scored.append((items, score))

        # ---- SORT ----
        scored.sort(key=lambda x: x[1], reverse=True)

        # ---- ROUTE SELECTION ----
        routes = self._select_routes(scored, context)

        return {"routes": routes}

    # =========================
    # GENERATE CANDIDATES
    # =========================
    def _generate_candidates(self, tops, bottoms, shoes, layers, accessories):

        candidates = []

        base_combos = list(itertools.product(tops, bottoms, shoes))

        # limit explosion
        random.shuffle(base_combos)
        base_combos = base_combos[:25]

        for top, bottom, shoe in base_combos:

            items = [top, bottom, shoe]

            # optional layering
            if layers and random.random() > 0.5:
                items.append(random.choice(layers))

            # optional accessory
            if accessories and random.random() > 0.5:
                items.append(random.choice(accessories))

            candidates.append(items)

        return candidates

    # =========================
    # ROUTE SELECTION
    # =========================
    def _select_routes(self, scored, context):

        if not scored:
            return []

        outputs = []

        route_defs = [
            ("safe", "Easy Win", 0),
            ("elevated", "Sharp Upgrade", 1),
            ("bold", "Statement Move", 2),
        ]

        for r_type, label, index in route_defs:

            idx = min(index, len(scored) - 1)
            items, score = scored[idx]

            aesthetic = self._build_aesthetic(items, context, r_type)
            description = self._build_description(r_type, context)

            outputs.append({
                "type": r_type,
                "label": label,
                "outfit": {
                    "items": items,
                    "score": round(score, 3),
                    "aesthetic": aesthetic,
                    "description": description
                }
            })

        return outputs

    # =========================
    # AESTHETIC BUILDER
    # =========================
    def _build_aesthetic(self, items, context, route_type):

        colors = [i.get("color", "").lower() for i in items if i.get("color")]

        if len(set(colors)) == 1:
            color_story = "monochrome"
        elif len(set(colors)) <= 2:
            color_story = "balanced"
        else:
            color_story = "contrast"

        vibe_map = {
            "safe": "clean_minimal",
            "elevated": "refined_structured",
            "bold": "expressive_statement"
        }

        return {
            "vibe": vibe_map.get(route_type),
            "color_story": color_story,
            "occasion": context.get("occasion")
        }

    # =========================
    # DESCRIPTION
    # =========================
    def _build_description(self, route_type, context):

        occasion = context.get("occasion", "your day")

        if route_type == "safe":
            return f"Reliable, clean, and effortless — perfect for {occasion}."

        if route_type == "elevated":
            return f"A sharper, more styled take — ideal when you want to stand out at {occasion}."

        if route_type == "bold":
            return f"Confident and expressive — this look makes a statement for {occasion}."

        return "Strong option."


# Singleton
outfit_engine = OutfitEngine()
