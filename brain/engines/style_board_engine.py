import random
from typing import Any, Dict, List

from brain.engines.palette_engine import palette_engine


class StyleBoardEngine:
    """
    🎯 Converts outfit → Pinterest-style board

    Output:
    - aesthetic
    - vibe
    - color story
    - layout (hero + supporting)
    - visual grouping
    """

    # =========================
    # MAIN ENTRY
    # =========================
    def build_board(self, outfit: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:

        items = outfit.get("items", [])
        if not items:
            return {}

        palette = palette_engine.select_palette({
            "event": context.get("occasion"),
            "microtheme": context.get("style_dna", {}).get("aesthetic")
        })

        aesthetic = self._derive_aesthetic(items, context)
        vibe = self._derive_vibe(context, aesthetic)
        color_story = self._build_color_story(items, palette)

        layout = self._build_layout(items)

        return {
            "aesthetic": aesthetic,
            "vibe": vibe,
            "color_story": color_story,
            "layout": layout,
            "items": items,
            "score": outfit.get("score", 0)
        }

    # =========================
    # AESTHETIC DETECTION
    # =========================
    def _derive_aesthetic(self, items, context):

        colors = [str(i.get("color", "")).lower() for i in items]

        if all(c in ["black", "white", "grey", "beige"] for c in colors):
            return "minimal luxury"

        if any(c in ["olive", "brown", "tan"] for c in colors):
            return "earthy luxury"

        if any(c in ["blue", "denim"] for c in colors):
            return "casual clean"

        if len(set(colors)) >= 3:
            return "statement street"

        return context.get("style_dna", {}).get("aesthetic", "modern classic")

    # =========================
    # VIBE GENERATION
    # =========================
    def _derive_vibe(self, context, aesthetic):

        occasion = (context.get("occasion") or "").lower()

        vibe_map = {
            "airport": "effortless transit",
            "office": "clean authority",
            "date": "soft confident",
            "party": "elevated bold",
        }

        if occasion in vibe_map:
            return vibe_map[occasion]

        if "luxury" in aesthetic:
            return "quiet luxury"

        return "everyday refined"

    # =========================
    # COLOR STORY
    # =========================
    def _build_color_story(self, items, palette):

        item_colors = [str(i.get("color", "")).lower() for i in items if i.get("color")]

        palette_colors = [c.lower() for c in palette.get("hex", [])]

        # mix item + palette
        combined = list(dict.fromkeys(item_colors + palette_colors))

        return combined[:5]

    # =========================
    # LAYOUT GENERATION
    # =========================
    def _build_layout(self, items):

        # 🎯 hero selection (visual anchor)
        hero = self._pick_hero(items)

        supporting = [i for i in items if i != hero]

        return {
            "hero": hero,
            "supporting": supporting,
            "composition": self._composition_type(len(items))
        }

    # =========================
    # HERO LOGIC
    # =========================
    def _pick_hero(self, items):

        priority_types = ["outerwear", "tops", "dresses"]

        for p in priority_types:
            for i in items:
                if p in str(i.get("type", "")).lower():
                    return i

        return items[0]

    # =========================
    # COMPOSITION TYPE
    # =========================
    def _composition_type(self, count):

        if count <= 3:
            return "minimal_grid"

        if count <= 5:
            return "balanced_board"

        return "editorial_spread"


# Singleton
style_board_engine = StyleBoardEngine()
