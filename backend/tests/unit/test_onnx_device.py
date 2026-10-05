import pytest

from researchhelp.retrieval.onnx_device import CPU, CUDA, active_provider, onnx_providers


def test_cpu_is_always_cpu():
    assert onnx_providers("cpu") == CPU


def test_auto_returns_a_valid_provider_list():
    assert onnx_providers("auto") in (CPU, CUDA)


def test_invalid_device_rejected():
    with pytest.raises(ValueError):
        onnx_providers("tpu")


def test_active_provider_unknown_for_non_onnx_objects():
    assert active_provider(object()) == "unknown"
