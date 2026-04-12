import base64
import requests
import time
import uuid
import threading
import hashlib
import cv2
import numpy as np
from queue import Queue
from fastapi import APIRouter
from pydantic import BaseModel, validator

router = APIRouter()

# =========================
# CONFIG
# =========================
RUNPOD_URL = "https://wvntzm71uikrla-11434.proxy.runpod.net//remove-bg/batch"
MAX_RETRIES = 3
TIMEOUT = 20
BATCH_SIZE = 4
JOB_TTL = 300  # 5 mins

# =========================
# CACHE
# =========================
_BG_CACHE = {}

def _hash_base64(b64):
    return hashlib.md5(b64.encode()).hexdigest()

# =========================
# PREPROCESS
# =========================
def _resize_if_needed(base64_str, max_size=1024):
    try:
        img_bytes = base64.b64decode(base64_str)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return base64_str

        h, w = img.shape[:2]
        if max(h, w) <= max_size:
            return base64_str

        scale = max_size / max(h, w)
        resized = cv2.resize(img, (int(w * scale), int(h * scale)))

        _, buffer = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return base64.b64encode(buffer).decode()

    except:
        return base64_str

# =========================
# REQUEST MODEL
# =========================
class BGRemoveRequest(BaseModel):
    image_base64: str

    @validator("image_base64")
    def validate_base64(cls, v):
        if not v or len(v) < 100:
            raise ValueError("Invalid image")
        return v

# =========================
# QUEUE SYSTEM
# =========================
task_queue = Queue()
jobs = {}

# =========================
# RUNPOD CALL
# =========================
def call_runpod_batch(images):
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            res = requests.post(
                RUNPOD_URL,
                json={"images": images},
                timeout=TIMEOUT
            )

            if res.ok:
                data = res.json()
                if "results" in data:
                    return data["results"]

            last_error = f"{res.status_code}: {res.text}"

        except Exception as e:
            last_error = str(e)

        time.sleep(1.5 ** attempt)

    raise Exception(f"RunPod failed: {last_error}")

# =========================
# WORKER
# =========================
def worker():
    while True:
        batch = []
        job_refs = []

        while len(batch) < BATCH_SIZE:
            job = task_queue.get()

            # cache check
            cache_key = _hash_base64(job["image"])
            if cache_key in _BG_CACHE:
                job["result"] = _BG_CACHE[cache_key]
                job["done"] = True
                task_queue.task_done()
                continue

            # preprocess
            img = _resize_if_needed(job["image"])

            batch.append(img)
            job_refs.append(job)

            if task_queue.empty():
                break

        if not batch:
            continue

        try:
            results = call_runpod_batch(batch)

            for job, result in zip(job_refs, results):
                job["result"] = result
                job["done"] = True

                cache_key = _hash_base64(job["image"])
                _BG_CACHE[cache_key] = result

        except Exception as e:
            for job in job_refs:
                job["error"] = str(e)
                job["done"] = True

        for _ in job_refs:
            task_queue.task_done()

# start worker
threading.Thread(target=worker, daemon=True).start()

# =========================
# CLEANUP THREAD (VERY IMPORTANT)
# =========================
def cleanup_worker():
    while True:
        now = time.time()
        to_delete = []

        for job_id, job in jobs.items():
            if now - job["created_at"] > JOB_TTL:
                to_delete.append(job_id)

        for jid in to_delete:
            del jobs[jid]

        time.sleep(30)

threading.Thread(target=cleanup_worker, daemon=True).start()

# =========================
# JOB MANAGEMENT
# =========================
def enqueue_job(image_base64):
    job_id = str(uuid.uuid4())

    job = {
        "id": job_id,
        "image": image_base64,
        "done": False,
        "result": None,
        "error": None,
        "created_at": time.time()
    }

    jobs[job_id] = job
    task_queue.put(job)

    return job_id


def get_job(job_id):
    job = jobs.get(job_id)

    if not job:
        return {"status": "not_found"}

    if not job["done"]:
        return {"status": "processing"}

    if job["error"]:
        return {
            "status": "completed",
            "image_base64": job["image"],  # fallback to original
            "fallback": True
        }

    return {
        "status": "completed",
        "image_base64": job["result"]
    }

# =========================
# ROUTES
# =========================
@router.post("/remove-bg")
def remove_bg(request: BGRemoveRequest):
    job_id = enqueue_job(request.image_base64)
    return {"success": True, "job_id": job_id}


@router.get("/remove-bg/status/{job_id}")
def check_status(job_id: str):
    return get_job(job_id)
