import subprocess
from fastapi import APIRouter

router = APIRouter()


def _gpu_metrics() -> dict:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=2.0,
        )
        rows = []
        for line in (out or "").splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            rows.append(
                {
                    "gpu_index": int(parts[0]),
                    "name": parts[1],
                    "memory_used_mb": int(float(parts[2])),
                    "memory_total_mb": int(float(parts[3])),
                    "utilization_percent": int(float(parts[4])),
                    "temperature_c": int(float(parts[5])),
                }
            )
        return {"available": True, "gpus": rows}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


@router.get("/metrics")
def ops_metrics():
    llm_metrics = {}
    bg_metrics = {}

    try:
        from services.llm_service import get_runtime_metrics

        llm_metrics = get_runtime_metrics()
    except Exception as exc:
        llm_metrics = {"error": str(exc)}

    try:
        from routers.bg_remover import get_bg_runtime_metrics

        bg_metrics = get_bg_runtime_metrics()
    except Exception as exc:
        bg_metrics = {"error": str(exc)}

    return {
        "success": True,
        "llm": llm_metrics,
        "background_queue": bg_metrics,
        "gpu": _gpu_metrics(),
    }
