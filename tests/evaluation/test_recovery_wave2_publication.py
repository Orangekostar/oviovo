from pathlib import Path

import pytest

from static_ovmap.module_validation.contracts import atomic_write_json
from static_ovmap.recovery_wave2.publication import verify_ready


def test_publication_rejects_pending_work_and_missing_primary_review(tmp_path):
    binding = {"output_root": str(tmp_path), "repository_root": str(tmp_path / "repo")}
    root = Path(binding["repository_root"]) / "artifacts/static_ovmap/recovery_wave2_v1"
    atomic_write_json(root / "completion.json", {"implementation_status": "PENDING"})
    with pytest.raises(ValueError, match="complete"):
        verify_ready(binding)
    atomic_write_json(root / "completion.json", {"implementation_status": "IMPLEMENTATION_COMPLETE"})
    with pytest.raises(FileNotFoundError, match="primary_review"):
        verify_ready(binding)
