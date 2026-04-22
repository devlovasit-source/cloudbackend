import io
import uuid
import asyncio
from typing import List, Dict

import torch
import numpy as np
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection

import mediapipe as mp

from services.bg_service import remove_bg_external
from services.r2_storage import R2Storage


# =========================
# INIT (LOAD ONCE)
# =========================
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose


# =========================
# QUERIES (INDIAN + WESTERN)
# =========================
ALL_QUERIES = [[
    "t-shirt", "shirt", "jacket", "hoodie",
    "dress", "jeans", "pants", "skirt",
    "saree", "lehenga", "kurta", "dupatta",
    "watch", "belt", "bracelet", "necklace", "earring"
]]


# =========================
# DETECTION (SINGLE PASS)
# =========================
def detect_items(image: Image.Image):
    inputs = processor(text=ALL_QUERIES, images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process(outputs, target_sizes=target_sizes)[0]

    detections = []
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        if score < 0.25:
            continue

        detections.append({
            "label": ALL_QUERIES[0][label],
            "bbox": box.tolist(),
            "score": float(score)
        })

    return detections


# =========================
# MEDIAPIPE REGIONS
# =========================
def get_regions(image_np):
    h, w, _ = image_np.shape
    regions = []

    with mp_face.FaceMesh(static_image_mode=True) as face:
        res = face.process(image_np)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0]
            x = int(lm.landmark[234].x * w)
            y = int(lm.landmark[234].y * h)

            regions.append({"label": "earring", "bbox": [x-40, y-40, x+40, y+40]})

    with mp_pose.Pose(static_image_mode=True) as pose:
        res = pose.process(image_np)
        if res.pose_landmarks:
            wrist = res.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            x = int(wrist.x * w)
            y = int(wrist.y * h)

            regions.append({"label": "watch", "bbox": [x-50, y-50, x+50, y+50]})

    return regions


# =========================
# FILTER
# =========================
def filter_boxes(detections, width, height):
    out = []
    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])

        if (x2 - x1) < 30 or (y2 - y1) < 30:
            continue

        area = (x2 - x1) * (y2 - y1)
        if area > 0.9 * width * height:
            continue

        out.append(d)

    return out


# =========================
# PROCESS ITEM
# =========================
async def process_crop(image, bbox, label, r2):
    x1, y1, x2, y2 = map(int, bbox)

    crop = image.crop((x1, y1, x2, y2))

    buf = io.BytesIO()
    crop.save(buf, format="JPEG")
    raw_bytes = buf.getvalue()

    loop = asyncio.get_event_loop()
    masked_bytes = await loop.run_in_executor(None, remove_bg_external, raw_bytes)

    file_id = str(uuid.uuid4())

    upload = r2.upload_wardrobe_images(
        file_id=file_id,
        raw_image_bytes=raw_bytes,
        masked_image_bytes=masked_bytes
    )

    return {
        "item_id": file_id,
        "label": label,
        "raw_url": upload["raw_image_url"],
        "masked_url": upload["masked_image_url"]
    }


# =========================
# MAIN PIPELINE
# =========================
async def run_hybrid_detection(image: Image.Image):
    width, height = image.size
    image_np = np.array(image)

    r2 = R2Storage()

    detections = detect_items(image)
    detections = filter_boxes(detections, width, height)

    # add mediapipe regions
    detections.extend(get_regions(image_np))

    tasks = [
        process_crop(image, d["bbox"], d["label"], r2)
        for d in detections
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    return [r for r in results if isinstance(r, dict)]
