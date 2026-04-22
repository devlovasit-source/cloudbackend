import io
import uuid
import asyncio
from typing import List, Dict

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

import mediapipe as mp

from services.bg_service import remove_bg_bytes
from services.r2_storage import R2Storage


# =========================
# CONFIG
# =========================
MAX_ITEMS = 5
RESIZE_LIMIT = 640


# =========================
# INIT MODEL (GROUNDING DINO TINY)
# =========================
device = torch.device("cpu")

processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
model = AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-tiny"
)

model.to(device)
model.eval()


TEXT_PROMPT = "shirt . pants . dress . saree . kurta . watch ."


# =========================
# RESIZE
# =========================
def resize_image(image: Image.Image):
    w, h = image.size
    if max(w, h) <= RESIZE_LIMIT:
        return image

    scale = RESIZE_LIMIT / max(w, h)
    return image.resize((int(w * scale), int(h * scale)))


# =========================
# DETECTION (DINO)
# =========================
def detect_items(image: Image.Image):
    inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    results = processor.post_process_object_detection(
        outputs,
        threshold=0.35,
        target_sizes=[image.size[::-1]]
    )[0]

    detections = []

    for score, label, box in zip(
        results["scores"],
        results["labels"],
        results["boxes"]
    ):
        detections.append({
            "label": processor.tokenizer.decode([label]).strip(),
            "bbox": box.tolist(),
            "score": float(score)
        })

    return detections


# =========================
# MEDIAPIPE
# =========================
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose


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
# IOU DEDUP
# =========================
def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter
    return inter / union if union else 0


def filter_and_limit(detections, width, height):
    filtered = []

    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])

        if (x2 - x1) < 30 or (y2 - y1) < 30:
            continue

        area = (x2 - x1) * (y2 - y1)
        if area > 0.9 * width * height:
            continue

        duplicate = False
        for f in filtered:
            if iou(f["bbox"], d["bbox"]) > 0.7:
                duplicate = True
                break

        if not duplicate:
            filtered.append(d)

    filtered.sort(key=lambda x: x["score"], reverse=True)
    return filtered[:MAX_ITEMS]


# =========================
# BATCH BG REMOVAL
# =========================
async def batch_bg(crops):
    return await asyncio.gather(
        *[asyncio.to_thread(remove_bg_bytes, c) for c in crops]
    )


# =========================
# MAIN PIPELINE
# =========================
async def run_hybrid_detection(image: Image.Image):
    image = resize_image(image)

    width, height = image.size
    image_np = np.array(image)

    r2 = R2Storage()

    # -------------------------
    # DETECTION
    # -------------------------
    detections = detect_items(image)
    detections = filter_and_limit(detections, width, height)

    # add mediapipe
    detections.extend(get_regions(image_np))

    # -------------------------
    # CROPS
    # -------------------------
    crops = []
    meta = []

    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])

        crop = image.crop((x1, y1, x2, y2))

        buf = io.BytesIO()
        crop.save(buf, format="JPEG")

        crops.append(buf.getvalue())
        meta.append(d)

    # -------------------------
    # BG REMOVAL (BATCH)
    # -------------------------
    masked_list = await batch_bg(crops)

    # -------------------------
    # UPLOAD (PARALLEL)
    # -------------------------
    async def upload_one(raw, masked, label):
        file_id = str(uuid.uuid4())

        upload = r2.upload_wardrobe_images(
            file_id=file_id,
            raw_image_bytes=raw,
            masked_image_bytes=masked
        )

        return {
            "item_id": file_id,
            "label": label,
            "raw_url": upload["raw_image_url"],
            "masked_url": upload["masked_image_url"]
        }

    results = await asyncio.gather(
        *[upload_one(r, m, meta[i]["label"]) for i, (r, m) in enumerate(zip(crops, masked_list))]
    )

    return results
