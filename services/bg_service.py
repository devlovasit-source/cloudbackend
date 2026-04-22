import io
import base64
import torch
import threading
import numpy as np
from PIL import Image
from transformers import AutoModelForImageSegmentation
from torchvision import transforms


# =========================
# INIT
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
model_lock = threading.Lock()

transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# =========================
# LOAD MODEL (ONCE)
# =========================
def load_model():
    global model

    if model is not None:
        return model

    with model_lock:
        if model is not None:
            return model

        model = AutoModelForImageSegmentation.from_pretrained(
            "briaai/RMBG-2.0",
            trust_remote_code=True
        )

        model.to(device)
        model.eval()

    return model


# =========================
# CORE FUNCTION (🔥 NEW)
# =========================
def remove_bg_bytes(image_bytes: bytes) -> bytes:
    """
    🔥 MAIN FUNCTION (used by hybrid pipeline)
    Input: raw image bytes
    Output: PNG bytes with alpha
    """

    try:
        model = load_model()

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = image.size

        input_tensor = transform_image(image).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(input_tensor)[-1].sigmoid().cpu()

        mask = preds[0].squeeze().numpy()
        mask = (mask > 0.5).astype("uint8") * 255

        mask_pil = Image.fromarray(mask).resize((w, h), Image.LANCZOS)

        output = image.copy()
        output.putalpha(mask_pil)

        buffer = io.BytesIO()
        output.save(buffer, format="PNG")

        return buffer.getvalue()

    except Exception:
        # fallback → return original
        return image_bytes


# =========================
# BASE64 ADAPTER (OPTIONAL)
# =========================
def remove_bg_base64(image_base64: str) -> str:
    try:
        image_bytes = base64.b64decode(image_base64.split(",")[-1])
        result_bytes = remove_bg_bytes(image_bytes)
        return base64.b64encode(result_bytes).decode()
    except Exception:
        return image_base64
