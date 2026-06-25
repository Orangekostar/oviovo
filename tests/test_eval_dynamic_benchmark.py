import json
import tempfile
from pathlib import Path
from scripts.eval_dynamic_benchmark import evaluate, load_event_log, load_frame_metrics


def test_evaluate_all_events_detected():
    """When ghost appears in detection window, event is detected."""
    events = [
        {
            "frame": 150, "type": "remove",
            "expected_judgment": "DISAPPEARED",
            "detection_window": [200, 350],
        }
    ]
    metrics = [
        {"frame_id": 100, "ghost_object_count": 0},
        {"frame_id": 200, "ghost_object_count": 0},
        {"frame_id": 210, "ghost_object_count": 1},  # ghost appears
        {"frame_id": 300, "ghost_object_count": 1},
    ]
    report = evaluate(events, metrics)
    assert report["detected"] == 1
    assert report["recall"] == 1.0
    assert report["per_event"][0]["detected"]
    assert report["per_event"][0]["detect_frame"] == 210


def test_event_missed_outside_window():
    """Ghost outside detection window is not counted as detection."""
    events = [
        {
            "frame": 150, "type": "remove",
            "expected_judgment": "DISAPPEARED",
            "detection_window": [200, 350],
        }
    ]
    metrics = [
        {"frame_id": 100, "ghost_object_count": 0},
        {"frame_id": 200, "ghost_object_count": 0},
        {"frame_id": 400, "ghost_object_count": 1},  # too late
    ]
    report = evaluate(events, metrics)
    assert report["detected"] == 0


def test_false_positive_detection():
    """Ghost outside any detection window counts as false positive."""
    events = [
        {
            "frame": 150, "type": "remove",
            "expected_judgment": "DISAPPEARED",
            "detection_window": [200, 350],
        }
    ]
    metrics = [
        {"frame_id": 100, "ghost_object_count": 1},  # before event!
        {"frame_id": 500, "ghost_object_count": 1},  # after window
    ]
    report = evaluate(events, metrics)
    assert report["false_positive_frames"] == 2


def test_move_detection():
    """MOVED event detected via moved_this_frame."""
    events = [
        {
            "frame": 750, "type": "move",
            "expected_judgment": "MOVED",
            "detection_window": [800, 950],
        }
    ]
    metrics = [
        {"frame_id": 700, "moved_this_frame": [], "ghost_object_count": 0},
        {"frame_id": 850, "moved_this_frame": [5], "ghost_object_count": 0},
    ]
    report = evaluate(events, metrics)
    assert report["detected"] == 1
