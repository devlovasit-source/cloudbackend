import base64
import os
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
RUNPOD_URL = os.getenv("RUNPOD_BG_BATCH_URL", "https://wvntzm71uikrla-11434.proxy.runpod.net/remove-bg/batch")
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
# ALPHA CHECK (🔥 NEW)
# =========================
def _has_alpha(base64_str):
    try:
        img_bytes = base64.b64decode(base64_str)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        return img is not None and len(img.shape) == 3 and img.shape[2] == 4
    except:
        return False


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
# RUNPOD CALL (🔥 FIXED)
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

                # 🔥 flexible parsing
                if "results" in data:
                    return data["results"]
                elif "images" in data:
                    return data["images"]
                elif "output" in data and "images" in data["output"]:
                    return data["output"]["images"]

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

            # 🔥 CACHE
            cache_key = _hash_base64(job["image"])
            if cache_key in _BG_CACHE:
                job["result"] = _BG_CACHE[cache_key]
                job["done"] = True
                task_queue.task_done()
                continue

            # 🔥 ALPHA SKIP
            if _has_alpha(job["image"]):
                job["result"] = job["image"]
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

                # 🔥 normalize output
                if not result.startswith("data:image"):
                    result = f"data:image/png;base64,{result}"

                job["result"] = result
                job["done"] = True

                cache_key = _hash_base64(job["image"])
                _BG_CACHE[cache_key] = result

        except Exception as e:
            for job in job_refs:
                job["error"] = f"runpod_failed: {str(e)}"
                job["done"] = True

        for _ in job_refs:
            task_queue.task_done()


# start worker
threading.Thread(target=worker, daemon=True).start()


# =========================
# CLEANUP THREAD
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
            "image_base64": job["image"],
            "fallback": True
        }

    return {
        "status": "completed",
        "image_base64": job["result"]
    }


def get_bg_runtime_metrics() -> dict:
    processing = 0
    completed = 0
    errored = 0
    for row in jobs.values():
        if not row.get("done"):
            processing += 1
        elif row.get("error"):
            errored += 1
        else:
            completed += 1

    return {
        "queue_size": int(task_queue.qsize()),
        "jobs_total": int(len(jobs)),
        "jobs_processing": int(processing),
        "jobs_completed": int(completed),
        "jobs_errored": int(errored),
        "cache_size": int(len(_BG_CACHE)),
        "batch_size": int(BATCH_SIZE),
        "timeout_seconds": int(TIMEOUT),
        "max_retries": int(MAX_RETRIES),
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


def remove_background_sync(image_base64: str):
    """
    Synchronous compatibility helper used by legacy routes.
    Returns the same shape expected by main.py and vision.py.
    """
    try:
        result_list = call_runpod_batch([_resize_if_needed(image_base64)])
        result = result_list[0] if isinstance(result_list, list) and result_list else ""
        if not isinstance(result, str) or not result.strip():
            return {"success": False, "bg_removed": False, "fallback_reason": "empty_runpod_response"}
        if result.startswith("data:image"):
            normalized = result
        else:
            normalized = f"data:image/png;base64,{result}"
        return {
            "success": True,
            "bg_removed": True,
            "image_base64": normalized,
            "fallback_reason": None,
        }
    except Exception as exc:
        return {
            "success": False,
            "bg_removed": False,
            "image_base64": image_base64,
            "fallback_reason": f"runpod_failed: {exc}",
        }
