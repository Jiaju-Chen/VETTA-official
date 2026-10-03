from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
from verl.utils.debug import performance


def test_cpu_memory_logging_does_not_block_actual_worker_calls(monkeypatch):
    monkeypatch.setattr(performance, "get_torch_device", lambda: SimpleNamespace())
    assert performance._get_current_mem_info() == ("N/A",) * 4
    called = []

    @performance.GPUMemoryLogger(role="cpu-test")
    def worker_call():
        called.append(True)
        return 7

    assert worker_call() == 7 and called == [True]
