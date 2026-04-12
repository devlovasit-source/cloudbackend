import base64
import requests
import time
import uuid
import threading
from queue import Queue
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, validator

router = APIRouter()

# =========================
# CONFIG
# =========================
RUNPOD_URL = "YOUR_RUNPOD_ENDPOINT/remove-bg/batch"
MAX_RETRIES = 3
TIMEOUT = 25
BATCH_SIZE = 4

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
# RUNPOD CALL (RETRY)
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

            if res.status_code == 200:
                return res.json()["results"]

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

        # collect batch
        while len(batch) < BATCH_SIZE:
            job = task_queue.get()
            batch.append(job["image"])
            job_refs.append(job)

            if task_queue.empty():
                break

        try:
            results = call_runpod_batch(batch)

            for job, result in zip(job_refs, results):
                job["result"] = result
                job["done"] = True

        except Exception as e:
            for job in job_refs:
                job["error"] = str(e)
                job["done"] = True

        for _ in job_refs:
            task_queue.task_done()

# start worker thread
threading.Thread(target=worker, daemon=True).start()

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
        "error": None
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
        return {"status": "failed", "error": job["error"]}

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
    return {
        "success": True,
        "job_id": job_id
    }


@router.get("/remove-bg/status/{job_id}")
def check_status(job_id: str):
    return get_job(job_id)
