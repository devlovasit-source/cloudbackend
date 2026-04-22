import io
import uuid
import asyncio
import base64
import os
from typing import List, Dict

import numpy as np
import requests
from PIL import Image
import mediapipe as mp

from services.bg_service import remove_bg_bytes
from services.r2_storage import R2Storage


# =========================
# CONFIG
# =========================
MAX_ITEMS = 5
RESIZE_LIMIT = 640

HF_TOKEN = os.getenv("HF_TOKEN")
HF_URL = "https://api-inference.huggingface.co/models/IDEA-Research/grounding-dino-tiny"

TEXT_PROMPT = "shirt . pants . dress . saree . kurta . watch . shoes . bag"


# =========================
# RESIZE
# =========================
def resize_image(image: Image.Image):
    w, h = image.size
    if max(w, h) <= RESIZE_LIMIT:
        return image

    scale = RESIZE_LIMIT / max(w, h)
    return image.resize((int(w * scale), int(h * scale)))


def image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


# =========================
# HF DETECTION
# =========================
def hf_detect(image: Image.Image):
    if not HF_TOKEN:
        print("[HF] token missing ❌")
        return []

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    image_b64 = image_to_base64(image)

    try:
        res = requests.post(
            HF_URL,
            headers=headers,
            json={
                "inputs": {
                    "image": image_b64,
                    "text": TEXT_PROMPT
                }
            },
            timeout=20
        )

        print("[HF STATUS]", res.status_code)

        if res.status_code != 200:
            print("[HF ERROR]", res.text)
            return []

        return res.json()

    except Exception as e:
        print("[HF EXCEPTION]", e)
        return []


def parse_detections(hf_output):
    detections = []

    for item in hf_output:
        box = item.get("box", {})

        detections.append({
            "label": item.get("label"),
            "bbox": [
                int(box.get("xmin", 0)),
                int(box.get("ymin", 0)),
                int(box.get("xmax", 0)),
                int(box.get("ymax", 0)),
            ],
            "score": float(item.get("score", 0))
        })

    return detections


# =========================
# MEDIAPIPE (ACCESSORIES BOOST)
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
            regions.append({"label": "earring", "bbox": [x-40, y-40, x+40, y+40], "score": 0.9})

    with mp_pose.Pose(static_image_mode=True) as pose:
        res = pose.process(image_np)
        if res.pose_landmarks:
            wrist = res.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            x = int(wrist.x * w)
            y = int(wrist.y * h)
            regions.append({"label": "watch", "bbox": [x-50, y-50, x+50, y+50], "score": 0.9})

    return regions


# =========================
# IOU + FILTER
# =========================
def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = ((box1[2]-box1[0])*(box1[3]-box1[1]) +
             (box2[2]-box2[0])*(box2[3]-box2[1]) - inter)

    return inter / union if union else 0


def filter_and_limit(detections, width, height):
    filtered = []

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]

        if (x2 - x1) < 30 or (y2 - y1) < 30:
            continue

        area = (x2 - x1) * (y2 - y1)
        if area > 0.9 * width * height:
            continue

        if any(iou(f["bbox"], d["bbox"]) > 0.7 for f in filtered):
            continue

        filtered.append(d)

    filtered.sort(key=lambda x: x["score"], reverse=True)
    return filtered[:MAX_ITEMS]


# =========================
# BG REMOVAL
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
    # HF DETECTION
    # -------------------------
    hf_output = hf_detect(image)
    detections = parse_detections(hf_output)

    # add mediapipe intelligence
    detections.extend(get_regions(image_np))

    detections = filter_and_limit(detections, width, height)

    if not detections:
        return []

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
    # BG REMOVAL
    # -------------------------
    masked_list = await batch_bg(crops)

    # -------------------------
    # UPLOAD
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
        *[
            upload_one(crops[i], masked_list[i], meta[i]["label"])
            for i in range(len(crops))
        ]
    )

    return results
