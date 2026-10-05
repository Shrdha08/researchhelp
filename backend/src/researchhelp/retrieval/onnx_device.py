"""Choose where the local ONNX models (embeddings, reranker) run.

``ONNX_DEVICE``: ``cpu``, ``cuda``, or ``auto`` (CUDA if onnxruntime-gpu and the CUDA DLLs are
available, otherwise CPU). On Windows the CUDA/cuDNN DLLs come from the ``nvidia-*`` pip wheels
(``uv sync --extra gpu``) and are loaded with ``onnxruntime.preload_dlls()``.

ONNX Runtime silently falls back to CPU when CUDA cannot be initialised, so callers should log
``active_provider(...)`` after loading a model.
"""

import logging
from functools import lru_cache

log = logging.getLogger(__name__)

CPU = ["CPUExecutionProvider"]
CUDA = ["CUDAExecutionProvider", "CPUExecutionProvider"]


@lru_cache
def onnx_providers(device: str = "auto") -> list[str]:
    device = device.lower()
    if device == "cpu":
        return CPU
    if device not in ("auto", "cuda"):
        raise ValueError(f"ONNX_DEVICE must be auto, cuda or cpu, not {device!r}")

    import onnxruntime as ort

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        if device == "cuda":
            raise RuntimeError(
                "ONNX_DEVICE=cuda but onnxruntime has no CUDA provider; run `uv sync --extra gpu`."
            )
        return CPU
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls(directory="")  # load CUDA/cuDNN DLLs shipped in nvidia-* wheels
    return CUDA


def active_provider(fastembed_model) -> str:
    """First execution provider of the model's ONNX session ("unknown" if not inspectable)."""
    inner = getattr(fastembed_model, "model", None)
    session = getattr(inner, "model", None)
    try:
        return session.get_providers()[0]
    except (AttributeError, IndexError):
        return "unknown"


def log_provider(name: str, fastembed_model, requested: list[str]) -> str:
    provider = active_provider(fastembed_model)
    if requested == CUDA and provider != "CUDAExecutionProvider":
        log.warning("%s: CUDA requested but running on %s", name, provider)
    else:
        log.info("%s running on %s", name, provider)
    return provider
