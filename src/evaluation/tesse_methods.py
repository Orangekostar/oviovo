"""Shared method labels for TESSE-CD causal snapshot artifacts."""

from __future__ import annotations


CAUSAL_SNAPSHOT_METHOD_LABELS = {
    "DUALMAP": frozenset({"DUALMAP", "DualMap"}),
    "DualMap": frozenset({"DUALMAP", "DualMap"}),
    "KHRONOS_OPEN": frozenset({"KHRONOS_OPEN", "KHRONOS", "Khronos"}),
    "KHRONOS_ORACLE": frozenset({"KHRONOS_ORACLE", "KHRONOS", "Khronos"}),
    "OVIV2": frozenset({"OVIV2"}),
    "PANOPTIC_SHARED": frozenset(
        {"PANOPTIC_SHARED", "Panoptic Mapping + shared masks"}
    ),
}


def snapshot_method_matches(method: str, snapshot_method: str) -> bool:
    return snapshot_method in CAUSAL_SNAPSHOT_METHOD_LABELS.get(method, frozenset())
