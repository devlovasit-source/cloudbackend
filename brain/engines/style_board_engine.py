import random
from typing import Any, Dict, List

from brain.engines.palette_engine import palette_engine


class StyleBoardEngine:
    """
    🔥 ELITE STYLE BOARD ENGINE (EDITORIAL)

    - Visual hierarchy (hero dominance)
    - Depth layering (foreground / mid / background)
    - Fashion-aware placement
    - Aesthetic intelligence (not basic rules)
    - Editorial composition types
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

        aesthetic = self._derive_aesthetic(items, context, palette)
        vibe = self._derive_vibe(context, aesthetic)
        color_story = self._build_color_story(items, palette)

        layout = self._build_editorial_layout(items, aesthetic)

        return {
            "aesthetic": aesthetic,
            "vibe": vibe,
            "color_story": color_story,
            "layout": layout,
            "items": items,
            "score": outfit.get("score", 0)
        }

    # =========================
    # 🎨 AESTHETIC INTELLIGENCE
    # =========================
    def _derive_aesthetic(self, items, context, palette):

        colors = [str(i.get("color", "")).lower() for i in items if i.get("color")]
        fits = [str(i.get("fit", "")).lower() for i in items if i.get("fit")]

        palette_tags = palette.get("tags", [])

        # 🔥 color logic
        if len(set(colors)) == 1:
            return "monochrome luxury"

        if len(set(colors)) == 2:
            return "balanced minimal"

        if len(set(colors)) >= 3:
            return "editorial contrast"

        # 🔥 silhouette logic
        if any("oversized" in f for f in fits):
            return "relaxed luxury"

        if any("tailored" in f for f in fits):
            return "structured refinement"

        # 🔥 palette influence
        if "luxury" in palette_tags:
            return "quiet luxury"

        return context.get("style_dna", {}).get("aesthetic", "modern refined")

    # =========================
    # 💫 VIBE
    # =========================
    def _derive_vibe(self, context, aesthetic):

        occasion = (context.get("occasion") or "").lower()

        vibe_map = {
            "airport": "effortless transit",
            "office": "quiet authority",
            "date": "soft allure",
            "party": "elevated presence",
        }

        if occasion in vibe_map:
            return vibe_map[occasion]

        if "luxury" in aesthetic:
            return "quiet luxury mood"

        return "refined everyday"

    # =========================
    # 🎨 COLOR STORY
    # =========================
    def _build_color_story(self, items, palette):

        item_colors = [str(i.get("color", "")).lower() for i in items if i.get("color")]
        palette_colors = [c.lower() for c in palette.get("hex", [])]

        combined = list(dict.fromkeys(item_colors + palette_colors))

        return combined[:5]

    # =========================
    # 🧠 EDITORIAL LAYOUT
    # =========================
    def _build_editorial_layout(self, items, aesthetic):

        hero = self._pick_hero(items)

        supporting = [i for i in items if i != hero]

        return {
            "composition": self._composition_type(len(items), aesthetic),
            "layers": self._build_layers(hero, supporting),
            "placements": self._build_placements(hero, supporting),
        }

    # =========================
    # 👑 HERO SELECTION
    # =========================
    def _pick_hero(self, items):

        priority = ["outerwear", "dress", "blazer", "jacket", "top"]

        for p in priority:
            for i in items:
                if p in str(i.get("type", "")).lower():
                    return i

        return items[0]

    # =========================
    # 🧱 LAYERING SYSTEM
    # =========================
    def _build_layers(self, hero, supporting):

        layers = {
            "foreground": [hero],
            "midground": [],
            "background": []
        }

        for item in supporting:
            t = str(item.get("type", "")).lower()

            if "shoes" in t or "footwear" in t:
                layers["foreground"].append(item)

            elif "accessory" in t or "bag" in t:
                layers["midground"].append(item)

            else:
                layers["background"].append(item)

        return layers

    # =========================
    # 📐 INTELLIGENT PLACEMENT
    # =========================
    def _build_placements(self, hero, supporting):

        placements = {}

        # 🔥 hero dominance
        placements[hero.get("id")] = {
            "x": 0.5,
            "y": 0.45,
            "scale": 1.2,
            "rotation": 0,
            "z": 3
        }

        # 🔥 supporting flow
        angles = [-25, -10, 10, 25]
        positions = [(0.2, 0.7), (0.8, 0.7), (0.3, 0.2), (0.7, 0.2)]

        for i, item in enumerate(supporting):

            placements[item.get("id")] = {
                "x": positions[i % len(positions)][0],
                "y": positions[i % len(positions)][1],
                "scale": round(random.uniform(0.6, 0.9), 2),
                "rotation": angles[i % len(angles)],
                "z": 2 if i < 2 else 1
            }

        return placements

    # =========================
    # 🧾 COMPOSITION TYPE
    # =========================
    def _composition_type(self, count, aesthetic):

        if "luxury" in aesthetic:
            return "editorial_focus"

        if count <= 3:
            return "minimal_grid"

        if count <= 5:
            return "balanced_editorial"

        return "magazine_spread"


# Singleton
style_board_engine = StyleBoardEngine()
