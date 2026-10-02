"""Report only the two measured cohorts in the metadata-bearing dataset spec."""

from static_ovmap.module_validation.contracts import atomic_write_json
from static_ovmap.recovery_wave2.light import METHODS
from static_ovmap.recovery_wave2.reporting import collect_matrix


def test_matrix_uses_fixed_cohorts_and_ignores_dataset_metadata(tmp_path):
    root, parent = tmp_path / "task", tmp_path / "parent"
    datasets = {"development": ["s1", "s2", "s3", "s4"],
        "replica": ["r" + str(i) for i in range(8)], "all_exposed": True,
        "schedule": "exact original completed frames"}
    metrics = dict(apall=.1, ap50=.2, ap25=.3, miou=.4, macc=.5)
    atomic_write_json(root / "freeze/receipt.json", {"light_package": {"id": "RW_B_D2"}, "nominated_map": None})
    atomic_write_json(root / "selection/final.json", {"map_inputs": []})
    for cohort in ("development", "replica"):
        methods = METHODS + (["RW_LIGHT_COMBO"] if cohort == "development" else [])
        for method in methods:
            atomic_write_json(root / "light/pools" / cohort / "BB00_NATIVE" / (method + ".json"),
                {"status": "COMPLETE", "scene_order": datasets[cohort], "identity": cohort + method, "metrics": metrics})
        atomic_write_json(parent / "pools" / cohort / "BB00_NATIVE/NATIVE_READOUT/OFFICIAL_CURRENT_CLASS.json",
            {"status": "COMPLETE", "scene_order": datasets[cohort], "identity": cohort + "native", "metrics": metrics})
    rows, pools = collect_matrix({"output_root": str(root), "parent_root": str(parent), "datasets": datasets})
    assert len(rows) == len(pools) == 19
    assert {row["cohort"] for row in rows} == {"development", "replica"}
    assert all(row["status"] == "COMPLETE" and row["scientific_status"] == "NO_NET_GAIN" for row in rows)
    assert {row["coverage"] for row in rows} == {"4/4", "8/8"}
    assert [row["cohort"] for row in rows if row["method"] == "RW_LIGHT_COMBO"] == ["development"]
