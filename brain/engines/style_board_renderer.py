import io
import random
import base64
from typing import Dict, Any, List

import requests
from PIL import Image, ImageFilter, ImageDraw, ImageFont

from routes.remove_bg import remove_background_sync


class StyleBoardRenderer:
    """
    🎨 EDITORIAL + INTELLIGENT STYLE BOARD RENDERER

    Combines:
    - background removal (real cutouts)
    - fashion-aware layout (stylist brain)
    - editorial composition (Zara/Pinterest feel)
    """

    CANVAS_SIZE = (1024, 1280)

    # =========================
    # MAIN ENTRY
    # =========================
    def render(self, board: Dict[str, Any]) -> bytes:

        canvas = self._create_background(board)

        layout = board.get("layout", {})
        hero = layout.get("hero")
        supporting = layout.get("supporting", [])

        all_items = [hero] + supporting if hero else supporting

        # 🔥 BACK LAYER FIRST (depth)
        self._place_fashion(canvas, all_items)

        # 🔥 TEXT LAST (on top)
        canvas = self._add_text(canvas, board)

        # 🔥 LIGHT FINISH
        canvas = self._add_light_gradient(canvas)

        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
        buffer.seek(0)

        return buffer.read()

    # =========================
    # BACKGROUND
    # =========================
    def _create_background(self, board):

        aesthetic = str(board.get("aesthetic", "")).lower()

        if "luxury" in aesthetic:
            base = (245, 238, 230)
        elif "street" in aesthetic:
            base = (25, 25, 25)
        else:
            base = (250, 250, 250)

        img = Image.new("RGB", self.CANVAS_SIZE, base)

        overlay = Image.new("RGBA", self.CANVAS_SIZE, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        for i in range(0, self.CANVAS_SIZE[1], 2):
            alpha = int(40 * (i / self.CANVAS_SIZE[1]))
            draw.rectangle([(0, i), (self.CANVAS_SIZE[0], i + 2)], fill=(255, 255, 255, alpha))

        img.paste(overlay, (0, 0), overlay)
        return img

    # =========================
    # FASHION LAYOUT ENGINE
    # =========================
    def _build_fashion_layout(self, items):

        layout = {
            "hero": None,
            "bottom": None,
            "shoes": None,
            "outerwear": None,
            "accessories": [],
        }

        for item in items:
            t = str(item.get("type", "")).lower()

            if any(k in t for k in ["shirt", "top", "tshirt", "blouse"]):
                layout["hero"] = item
            elif any(k in t for k in ["jeans", "pants", "trousers", "skirt"]):
                layout["bottom"] = item
            elif any(k in t for k in ["shoe", "sneaker", "heel"]):
                layout["shoes"] = item
            elif any(k in t for k in ["jacket", "coat", "blazer"]):
                layout["outerwear"] = item
            else:
                layout["accessories"].append(item)

        return layout

    def _place_fashion(self, canvas, items):

        layout = self._build_fashion_layout(items)

        # 🔥 OUTERWEAR (BACK LAYER)
        if layout["outerwear"]:
            img = self._prepare_image(layout["outerwear"], 550)
            img = self._add_shadow(img)
            canvas.paste(img, (220, 100), img)

        # 🔥 HERO
        if layout["hero"]:
            img = self._prepare_image(layout["hero"], 520)
            img = self._add_shadow(img)
            canvas.paste(img, (250, 120), img)

        # 🔥 BOTTOM
        if layout["bottom"]:
            img = self._prepare_image(layout["bottom"], 420)
            img = self._add_shadow(img)
            canvas.paste(img, (300, 420), img)

        # 🔥 SHOES (ANCHOR)
        if layout["shoes"]:
            img = self._prepare_image(layout["shoes"], 260)
            img = self._add_shadow(img)
            canvas.paste(img, (380, 820), img)

        # 🔥 ACCESSORIES
        self._place_accessories(canvas, layout["accessories"])

    def _place_accessories(self, canvas, items):

        zones = [
            (80, 200),
            (750, 250),
            (100, 900),
            (780, 950),
        ]

        for i, item in enumerate(items[:4]):
            img = self._prepare_image(item, 160)
            img = self._add_shadow(img)

            x, y = zones[i % len(zones)]

            x += random.randint(-20, 20)
            y += random.randint(-20, 20)

            canvas.paste(img, (x, y), img)

    # =========================
    # IMAGE LOADING + BG REMOVAL
    # =========================
    def _load_image(self, item):

        url = item.get("image_url")

        if not url:
            return Image.new("RGBA", (200, 200), (0, 0, 0, 0))

        try:
            res = requests.get(url, timeout=5)

            # convert to base64
            b64 = base64.b64encode(res.content).decode()

            # 🔥 CALL YOUR BG SERVICE
            result = remove_background_sync(b64)

            if result.get("success") and result.get("bg_removed"):
                clean_b64 = result["image_base64"].split(",")[-1]
            else:
                clean_b64 = b64

            img_bytes = base64.b64decode(clean_b64)
            return Image.open(io.BytesIO(img_bytes)).convert("RGBA")

        except Exception:
            return Image.new("RGBA", (200, 200), (0, 0, 0, 0))

    def _prepare_image(self, item, target_width):

        img = self._load_image(item)

        w, h = img.size
        ratio = target_width / w
        img = img.resize((target_width, int(h * ratio)))

        img = img.rotate(random.uniform(-3, 3), expand=True)

        return img

    # =========================
    # SHADOW
    # =========================
    def _add_shadow(self, img):

        shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))

        for x in range(img.size[0]):
            for y in range(img.size[1]):
                shadow.putpixel((x, y), (0, 0, 0, 80))

        shadow = shadow.filter(ImageFilter.GaussianBlur(15))

        base = Image.new("RGBA", img.size, (0, 0, 0, 0))
        base.paste(shadow, (10, 10), shadow)
        base.paste(img, (0, 0), img)

        return base

    # =========================
    # LIGHTING
    # =========================
    def _add_light_gradient(self, img):

        overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))

        for y in range(img.size[1]):
            alpha = int(60 * (1 - y / img.size[1]))
            for x in range(img.size[0]):
                overlay.putpixel((x, y), (255, 255, 255, alpha))

        img = img.convert("RGBA")
        img.paste(overlay, (0, 0), overlay)

        return img.convert("RGB")

    # =========================
    # TYPOGRAPHY
    # =========================
    def _add_text(self, img, board):

        draw = ImageDraw.Draw(img)

        title = str(board.get("vibe", "STYLE")).upper()
        subtitle = str(board.get("aesthetic", "")).title()

        try:
            title_font = ImageFont.truetype("PlayfairDisplay-Bold.ttf", 64)
            sub_font = ImageFont.truetype("Montserrat-Regular.ttf", 28)
        except:
            title_font = ImageFont.load_default()
            sub_font = ImageFont.load_default()

        draw.text((60, 80), title, fill=(20, 20, 20), font=title_font)
        draw.text((60, 150), subtitle, fill=(80, 80, 80), font=sub_font)

        return img


# Singleton
style_board_renderer = StyleBoardRenderer()
