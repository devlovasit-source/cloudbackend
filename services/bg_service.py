import requests
import os

HF_TOKEN = os.getenv("HF_TOKEN")

HF_BG_URL = "https://api-inference.huggingface.co/models/briaai/RMBG-2.0"


def remove_bg_bytes(image_bytes: bytes) -> bytes:
    if not HF_TOKEN:
        print("[BG] HF token missing ❌")
        return image_bytes

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/octet-stream"
    }

    try:
        res = requests.post(
            HF_BG_URL,
            headers=headers,
            data=image_bytes,
            timeout=30
        )

        print("[BG STATUS]", res.status_code)

        if res.status_code != 200:
            print("[BG ERROR]", res.text)
            return image_bytes

        return res.content

    except Exception as e:
        print("[BG EXCEPTION]", e)
        return image_bytes
