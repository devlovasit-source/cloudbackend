from typing import Tuple
import re


class ColorNormalizer:
    """
    🔥 COLOR NORMALIZATION ENGINE

    Converts:
    - hex → color name
    - rgb → color name (optional future)
    - keeps names unchanged

    Used by:
    - style_scorer
    - palette matching
    """

    # =========================
    # MAIN ENTRY
    # =========================
    def normalize(self, color: str) -> str:
        if not color:
            return ""

        color = color.strip().lower()

        # already a name
        if not color.startswith("#"):
            return color

        rgb = self._hex_to_rgb(color)
        if not rgb:
            return color

        return self._map_rgb_to_name(rgb)

    # =========================
    # HEX → RGB
    # =========================
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int] | None:
        hex_color = hex_color.replace("#", "")

        if len(hex_color) == 3:
            hex_color = "".join([c * 2 for c in hex_color])

        if len(hex_color) != 6 or not re.match(r"[0-9a-f]{6}", hex_color):
            return None

        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

        return (r, g, b)

    # =========================
    # RGB → COLOR NAME
    # =========================
    def _map_rgb_to_name(self, rgb: Tuple[int, int, int]) -> str:
        r, g, b = rgb

        # 🔥 simple but effective buckets

        if r < 50 and g < 50 and b < 50:
            return "black"

        if r > 200 and g > 200 and b > 200:
            return "white"

        if abs(r - g) < 20 and abs(g - b) < 20:
            return "grey"

        # dominant channels
        if r > g and r > b:
            if g > 100:
                return "orange"
            return "red"

        if g > r and g > b:
            return "green"

        if b > r and b > g:
            return "blue"

        # mixed tones
        if r > 150 and g > 120 and b < 100:
            return "beige"

        if r > 120 and g < 80 and b > 120:
            return "purple"

        return "unknown"


# Singleton
color_normalizer = ColorNormalizer()
