import random
from typing import Any, Dict, List

from brain.engines.style_graph_engine import style_graph_engine
from brain.engines.style_rules_engine import style_engine
from brain.engines.palette_engine import palette_engine


class OutfitEngine:
    """
    🔥 ELITE V3 — STYLIST SYSTEM

    - Route-based styling (safe / elevated / bold)
    - Body-aware (StyleRulesEngine)
    - Palette-first decisions
    - Graph compatibility
    - Aesthetic + description included
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

        # ---- ENGINES ----
        graph = style_graph_engine.build_graph({
            "tops": tops,
            "bottoms": bottoms,
            "shoes": shoes
        })

        rules = style_engine.get_scoring_rules(
            context.get("style_dna", {}),
            context
        )

        palette = palette_engine.select_palette({
            "event": context.get("occasion"),
            "microtheme": context.get("style_dna", {}).get("aesthetic")
        })

        palette_colors = [c.lower() for c in palette.get("hex", [])]

        # ---- ROUTES ----
        routes = self._build_routes()

        outputs = []

        for route in routes:

            top = self._pick_by_strategy(tops, route, rules, palette_colors)
            bottom = self._best_match(top, bottoms, graph, rules, palette_colors)
            shoe = self._best_match(top, shoes, graph, rules, palette_colors)

            layer = self._best_optional(layers, rules, palette_colors)
            accessory = self._best_optional(accessories, rules, palette_colors)

            items = [x for x in [top, bottom, shoe, layer, accessory] if x]

            score = self._score_outfit(items, graph, rules, palette_colors)
            aesthetic = self._build_aesthetic(items, context, palette, route)
            description = self._build_route_description(route, aesthetic, context)

            outputs.append({
                "type": route["type"],
                "label": route["label"],
                "outfit": {
                    "items": items,
                    "score": round(score, 3),
                    "aesthetic": aesthetic,
                    "description": description,
                    "palette": palette.get("name")
                }
            })

        outputs.sort(key=lambda x: x["outfit"]["score"], reverse=True)

        return {"routes": outputs}

    # =========================
    # ROUTES
    # =========================
    def _build_routes(self):
        return [
            {"type": "safe", "label": "Easy Win"},
            {"type": "elevated", "label": "Sharp Upgrade"},
            {"type": "bold", "label": "Statement Move"},
        ]

    # =========================
    # STRATEGY PICK
    # =========================
    def _pick_by_strategy(self, items, route, rules, palette_colors):

        ranked = self._rank_items(items, rules, palette_colors)

        if not ranked:
            return None

        if route["type"] == "safe":
            return ranked[0]

        if route["type"] == "elevated":
            return ranked[min(1, len(ranked)-1)]

        if route["type"] == "bold":
            return ranked[min(2, len(ranked)-1)]

        return ranked[0]

    # =========================
    # RANK ITEMS
    # =========================
    def _rank_items(self, items, rules, palette_colors):

        scored = []

        for item in items:
            score = 0
            text = str(item).lower()

            if item.get("color", "").lower() in palette_colors:
                score += 1

            if item.get("color", "").lower() in rules["preferred_colors"]:
                score += 0.5

            if any(k in text for k in rules["preferred_keywords"]):
                score += 1

            if item.get("type", "").lower() in rules["avoided_items"]:
                score -= 2

            scored.append((item, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in scored]

    # =========================
    # BEST MATCH
    # =========================
    def _best_match(self, anchor, candidates, graph, rules, palette_colors):

        if not anchor or not candidates:
            return None

        scored = []

        for c in candidates:
            score = 0

            score += style_graph_engine.pair_weight(
                graph,
                anchor.get("id"),
                c.get("id")
            )

            if c.get("color", "").lower() in palette_colors:
                score += 0.5

            if any(k in str(c).lower() for k in rules["preferred_keywords"]):
                score += 0.5

            scored.append((c, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    # =========================
    # OPTIONAL ITEMS
    # =========================
    def _best_optional(self, items, rules, palette_colors):
        if not items:
            return None

        ranked = self._rank_items(items, rules, palette_colors)
        return ranked[0] if random.random() > 0.3 else ranked[min(1, len(ranked)-1)]

    # =========================
    # SCORING
    # =========================
    def _score_outfit(self, items, graph, rules, palette_colors):

        score = 0

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                score += style_graph_engine.pair_weight(
                    graph,
                    items[i].get("id"),
                    items[j].get("id")
                )

        score += sum(
            0.4 for i in items
            if i.get("color", "").lower() in palette_colors
        )

        for i in items:
            text = str(i).lower()

            if any(k in text for k in rules["preferred_keywords"]):
                score += 0.3

            if i.get("type", "").lower() in rules["avoided_items"]:
                score -= 1

        return score

    # =========================
    # AESTHETIC
    # =========================
    def _build_aesthetic(self, items, context, palette, route):

        colors = [i.get("color", "").lower() for i in items if i.get("color")]

        if len(set(colors)) == 1:
            color_story = "monochrome"
        elif len(set(colors)) <= 2:
            color_story = "balanced"
        else:
            color_story = "contrast"

        base_vibe = {
            "safe": "clean_minimal",
            "elevated": "structured_refined",
            "bold": "expressive_statement"
        }

        return {
            "vibe": base_vibe.get(route["type"]),
            "color_story": color_story,
            "palette_name": palette.get("name"),
            "palette_tags": palette.get("tags", [])
        }

    # =========================
    # DESCRIPTION
    # =========================
    def _build_route_description(self, route, aesthetic, context):

        occasion = context.get("occasion", "day out")

        if route["type"] == "safe":
            return f"This is your no-fail look. Clean, balanced, and perfect for {occasion}."

        if route["type"] == "elevated":
            return f"This feels sharper. More intentional, more styled — ideal for {occasion}."

        if route["type"] == "bold":
            return f"This one stands out. Strong presence, high impact — if you're in the mood."

        return "Solid option."


# Singleton
outfit_engine = OutfitEngine()
