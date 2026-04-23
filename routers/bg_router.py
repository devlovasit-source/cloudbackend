import base64
from fastapi import APIRouter, HTTPException
<<<<<<< HEAD
from pydantic import BaseModel, validator
=======
from pydantic import BaseModel, field_validator
>>>>>>> ba59b6b (Fix routing imports, Pydantic v2 validators, chat cache thread safety, and auth error handling)

from services.bg_service import remove_bg_bytes

router = APIRouter(prefix="/bg", tags=["background-removal"])


# =========================
# REQUEST MODEL
# =========================
class BGRemoveRequest(BaseModel):
    image_base64: str

<<<<<<< HEAD
    @validator("image_base64")
=======
    @field_validator("image_base64")
    @classmethod
>>>>>>> ba59b6b (Fix routing imports, Pydantic v2 validators, chat cache thread safety, and auth error handling)
    def validate_base64(cls, v):
        if not v or len(v) < 100:
            raise ValueError("Invalid image data")
        return v


# =========================
# ROUTE (SYNC)
# =========================
@router.post("/remove")
def remove_background(request: BGRemoveRequest):
    try:
        # decode
        image_bytes = base64.b64decode(request.image_base64.split(",")[-1])

        # process (🔥 core service)
        result_bytes = remove_bg_bytes(image_bytes)

        # encode response
        result_base64 = base64.b64encode(result_bytes).decode()

        return {
            "success": True,
            "image_base64": result_base64
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BG removal failed: {e}")
