import os
from typing import List

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


# =========================
# 🔥 SINGLETON MODEL
# =========================
_model = None

_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2"
)


def _get_model():
    global _model

    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed")

    if _model is None:
        print(f"[embedding] loading model: {_MODEL_NAME}")
        _model = SentenceTransformer(_MODEL_NAME)

    return _model


# =========================
# TEXT BUILDING
# =========================
def _build_text(data: dict) -> str:
    """
    Convert metadata → flat text string
    """

    category = str(data.get("category") or "")
    sub_category = str(data.get("sub_category") or "")
    color = str(data.get("color_code") or "")
    pattern = str(data.get("pattern") or "")
    style = str(data.get("style") or "")

    occasions_raw = data.get("occasions", [])
    if isinstance(occasions_raw, list):
        occasions = " ".join(str(x) for x in occasions_raw if x)
    else:
        occasions = str(occasions_raw or "")

    text = " ".join([
        category,
        sub_category,
        color,
        pattern,
        style,
        occasions
    ]).strip()

    return text


# =========================
# 🔥 CORE ENCODERS
# =========================
def encode_text(text: str) -> List[float]:
    """
    Generic text embedding
    """
    try:
        if not text:
            return []

        model = _get_model()
        vector = model.encode(text)

        return vector.tolist()

    except Exception as e:
        print("[embedding] encode_text error:", e)
        return []


def encode_metadata(data: dict) -> List[float]:
    """
    Main function used across system
    """
    try:
        text = _build_text(data)
        return encode_text(text)

    except Exception as e:
        print("[embedding] encode_metadata error:", e)
        return []


# =========================
# OPTIONAL SERVICE WRAPPER
# =========================
class EmbeddingService:
    def encode_text(self, text: str) -> List[float]:
        return encode_text(text)

    def encode_metadata(self, data: dict) -> List[float]:
        return encode_metadata(data)


# 🔥 singleton instance (important for reuse)
embedding_service = EmbeddingService()
