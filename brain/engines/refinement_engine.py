from typing import Dict, Any, List
import copy


class RefinementEngine:
    """
    🔥 ADVANCED REFINEMENT ENGINE

    Capabilities:
    - Chip-driven transformations
    - Style DNA alignment
    - Memory-aware personalization
    - Wardrobe-based replacement (real items)
    - Safe fallbacks
    """

    # =========================
    # MAIN ENTRY
    # =========================
    def apply(self, outfits: List[Dict], context: Dict[str, Any]) -> List[Dict]:

        if not outfits:
            return outfits

        mode = context.get("refinement")
        style_dna = context.get("style_dna", {}) or {}
        memory = context.get("user_memory", {}) or {}
        wardrobe = context.get("wardrobe", []) or []

        refined_outfits = []

        for outfit in outfits:
            new_outfit = copy.deepcopy(outfit)
            items = new_outfit.get("items", [])

            # -------------------------
            # 1. CHIP-BASED TRANSFORMATION
            # -------------------------
            if mode:
                items = self._apply_mode(items, mode, context)

            # -------------------------
            # 2. STYLE DNA ALIGNMENT
            # -------------------------
            items = self._apply_style_dna(items, style_dna)

            # -------------------------
            # 3. MEMORY ALIGNMENT
            # -------------------------
            items = self._apply_memory(items, memory)

            # -------------------------
            # 4. WARDROBE SWAP (REAL ITEMS)
            # -------------------------
            items = self._apply_wardrobe_swap(items, wardrobe, context)

            new_outfit["items"] = items
            new_outfit["refined"] = mode or "auto"

            refined_outfits.append(new_outfit)

        return refined_outfits

    # =========================
    # CHIP MODE LOGIC
    # =========================
    def _apply_mode(self, items, mode, context):

        if mode == "sharp":
            return self._make_sharp(items)

        if mode == "relaxed":
            return self._make_relaxed(items)

        if mode == "bold":
            return self._make_bold(items)

        if mode == "minimal":
            return self._make_minimal(items)

        return items

    # -------------------------
    # SHARP
    # -------------------------
    def _make_sharp(self, items):
        new_items = []

        for i in items:
            t = str(i.get("type", "")).lower()

            if "hoodie" in t or "tshirt" in t:
                i["preferred_replacement"] = "shirt"
                i["style"] = "structured"

            if "sneaker" in t:
                i["preferred_replacement"] = "loafers"
                i["style"] = "formal"

            new_items.append(i)

        return new_items

    # -------------------------
    # RELAXED
    # -------------------------
    def _make_relaxed(self, items):
        new_items = []

        for i in items:
            t = str(i.get("type", "")).lower()

            if "shirt" in t:
                i["preferred_replacement"] = "tshirt"
                i["style"] = "casual"

            if "loafers" in t:
                i["preferred_replacement"] = "sneakers"
                i["style"] = "casual"

            new_items.append(i)

        return new_items

    # -------------------------
    # BOLD
    # -------------------------
    def _make_bold(self, items):
        for i in items:
            i["pattern"] = "statement"
            i["color"] = i.get("color") or "red"
        return items

    # -------------------------
    # MINIMAL
    # -------------------------
    def _make_minimal(self, items):
        for i in items:
            i["pattern"] = "solid"
            if i.get("color") not in ["black", "white", "beige", "grey"]:
                i["color"] = "neutral"
        return items

    # =========================
    # STYLE DNA ALIGNMENT
    # =========================
    def _apply_style_dna(self, items, dna):

        if not dna:
            return items

        preferred_styles = dna.get("preferred_styles", [])
        preferred_colors = dna.get("preferred_colors", [])

        for i in items:
            if preferred_styles:
                if i.get("style") not in preferred_styles:
                    i["style_boost"] = False
                else:
                    i["style_boost"] = True

            if preferred_colors:
                if i.get("color") in preferred_colors:
                    i["color_boost"] = True

        return items

    # =========================
    # MEMORY ALIGNMENT
    # =========================
    def _apply_memory(self, items, memory):

        signals = memory.get("memory_signals", {})

        liked_styles = signals.get("preferred_styles", [])
        disliked_items = signals.get("disliked_items", [])

        new_items = []

        for i in items:
            t = str(i.get("type", "")).lower()

            if t in disliked_items:
                i["avoid"] = True

            if i.get("style") in liked_styles:
                i["memory_boost"] = True

            new_items.append(i)

        return new_items

    # =========================
    # WARDROBE SWAP (REAL)
    # =========================
    def _apply_wardrobe_swap(self, items, wardrobe, context):

        if not wardrobe:
            return items

        swapped = []

        for i in items:

            replacement_type = i.get("preferred_replacement")

            if not replacement_type:
                swapped.append(i)
                continue

            candidate = self._find_best_item(
                replacement_type,
                wardrobe,
                context
            )

            if candidate:
                swapped.append(candidate)
            else:
                swapped.append(i)

        return swapped

    def _find_best_item(self, target_type, wardrobe, context):

        candidates = [
            w for w in wardrobe
            if target_type in str(w.get("type", "")).lower()
        ]

        if not candidates:
            return None

        # -------------------------
        # SIMPLE SCORING (CAN UPGRADE)
        # -------------------------
        def score(item):
            s = 0

            if item.get("color") in ["black", "white", "beige"]:
                s += 1

            if item.get("style") in ["formal", "minimal"]:
                s += 1

            return s

        candidates.sort(key=score, reverse=True)

        return candidates[0]


# singleton
refinement_engine = RefinementEngine()
