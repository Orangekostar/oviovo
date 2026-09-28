import importlib.util
import os
from pathlib import Path

from src.static_ovmap.replica_transfer.parallel import worker_config


def test_hardware_only_clone():
    original = {"identity": "abc123", "runtime": {"cuda_device": "2", "precision": "float32"}, "gpu_lock": "/tmp/.visual-gpu-2.lock", "temperatures": {"N0": 0.01}, "budget": 200}
    clone = worker_config(original, 0)
    assert original["runtime"] == {"cuda_device": "2", "precision": "float32"}
    assert clone["runtime"] == {"cuda_device": "0", "precision": "float32", "ipc_prefix": "replica_abc123"}
    assert clone["temperatures"] == original["temperatures"]
    assert clone["budget"] == 200
    assert clone["gpu_lock"] == "/tmp/.visual-gpu-0.lock"


def test_pinned_ipc_namespace_isolation():
    path = Path("/home/ww/crove/ovimap-module-validation-upstream/scripts/ipc_utils.py")
    spec = importlib.util.spec_from_file_location("test_replica_ipc", path)
    ipc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ipc)
    from scripts.evaluation.replica_isolated_mapper import isolate

    slots = []
    attached = {}
    try:
        isolate(ipc, f"replica_test_{os.getpid()}_a")
        a = ipc.create_shm_slots(2, 3, 1)
        slots.extend(a)
        a[0][2][:] = 7
        isolate(ipc, f"replica_test_{os.getpid()}_b")
        b = ipc.create_shm_slots(2, 3, 1)
        slots.extend(b)
        b[0][2][:] = 9
        attached = ipc.attach_shm_slots([a[0][0]], 2, 3)
        assert (attached[0][1] == 7).all()
        assert (b[0][2] == 9).all()
        assert a[0][0] != b[0][0]
    finally:
        for shm, _ in attached.values():
            shm.close()
        ipc.cleanup_shm_slots(slots)
