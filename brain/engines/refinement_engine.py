
from typing import Dict, Any, List
import copy

from brain.engines.wardrobe_selector import wardrobe_selector


class RefinementEngine:
    """
    🔥 FINAL ELITE REFINEMENT ENGINE

    Responsibilities:
    ✔ apply chip transformations
    ✔ align with style DNA
    ✔ align with memory
    ✔ delegate item selection to wardrobe_selector

    DOES NOT:
    ❌ perform scoring
    ❌ rank candidates manually
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

        refined_outfits = []

        for outfit in outfits:
            new_outfit = copy.deepcopy(outfit)
            items = new_outfit.get("items", [])

            # -------------------------
            # 1. CHIP TRANSFORMATION
            # -------------------------
            if mode:
                items = self._apply_mode(items, mode)

            # -------------------------
            # 2. STYLE DNA ALIGNMENT
            # -------------------------
            items = self._apply_style_dna(items, style_dna)

            # -------------------------
            # 3. MEMORY ALIGNMENT
            # -------------------------
            items = self._apply_memory(items, memory)

            # -------------------------
            # 4. REAL WARDROBE SWAP
            # -------------------------
            items = self._apply_wardrobe_swap(items, context)

            new_outfit["items"] = items
            new_outfit["refined"] = mode or "auto"

            refined_outfits.append(new_outfit)

        return refined_outfits

    # =========================
    # CHIP MODE
    # =========================
    def _apply_mode(self, items, mode):

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
        for i in items:
            t = str(i.get("type", "")).lower()

            if "hoodie" in t or "tshirt" in t:
                i["preferred_replacement"] = "shirt"
                i["style"] = "structured"

            if "sneaker" in t:
                i["preferred_replacement"] = "loafers"
                i["style"] = "formal"

        return items

    # -------------------------
    # RELAXED
    # -------------------------
    def _make_relaxed(self, items):
        for i in items:
            t = str(i.get("type", "")).lower()

            if "shirt" in t:
                i["preferred_replacement"] = "tshirt"
                i["style"] = "casual"

            if "loafers" in t:
                i["preferred_replacement"] = "sneakers"
                i["style"] = "casual"

        return items

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
    # STYLE DNA
    # =========================
    def _apply_style_dna(self, items, dna):

        if not dna:
            return items

        preferred_styles = dna.get("preferred_styles", [])
        preferred_colors = dna.get("preferred_colors", [])

        for i in items:
            if preferred_styles:
                i["style_boost"] = i.get("style") in preferred_styles

            if preferred_colors:
                i["color_boost"] = i.get("color") in preferred_colors

        return items

    # =========================
    # MEMORY
    # =========================
    def _apply_memory(self, items, memory):

        signals = memory.get("memory_signals", {})

        liked_styles = signals.get("preferred_styles", [])
        disliked_items = signals.get("disliked_items", [])

        for i in items:
            t = str(i.get("type", "")).lower()

            if t in disliked_items:
                i["avoid"] = True

            if i.get("style") in liked_styles:
                i["memory_boost"] = True

        return items

    # =========================
    # WARDROBE SWAP (FIXED)
    # =========================
    def _apply_wardrobe_swap(self, items, context):

        swapped = []

        for i in items:

            replacement_type = i.get("preferred_replacement")

            if not replacement_type:
                swapped.append(i)
                continue

            # 🔥 USE CENTRAL SELECTOR
            candidate = wardrobe_selector.find_best_match(
                replacement_type,
                context,
                reference_embedding=i.get("embedding")
            )

            if candidate:
                swapped.append(candidate)
            else:
                swapped.append(i)

        return swapped


# singleton
refinement_engine = RefinementEngine()
