# Reconstruct or resume

Dense predictions reuse their exact immutable N0 geometry and ranks.

```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python -m src.static_ovmap.composition_study.reconstruction --labels artifacts/static_ovmap/complementary_composition_v1/labels/compose_cal/scene0056_00/N0.json --output /mnt/shared/ww/ovimap-complementary-composition-v1/reconstruction/scene0056_00/N0
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_composition_study.py --resolved-config /mnt/shared/ww/ovimap-complementary-composition-v1/attempt_001/resolved_config.json --phase all
```

Use each labels/<role>/<scene>/<method>.json for the corresponding exact prediction; hashes and record keys are checked. External inputs are listed in external_artifacts.json. No data/model download is part of reconstruction.
