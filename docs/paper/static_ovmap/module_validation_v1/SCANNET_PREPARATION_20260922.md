# ScanNet data preparation — 2026-09-22

Status: preparation code implemented and tested; public split metadata downloaded; restricted raw downloads **not started**, pending confirmation of the user's existing ScanNet authorization and terms agreement. The independent-scene S/G/Q experiment driver remains incomplete. This change does not supply new scientific measurements.

Data root: `/mnt/shared/ww/ovimap-module-validation-v1/data/scannet`.

The preparation CLI reads literal release constants from `/home/ww/getscannet.py` without executing its full-release branch, interactive terms prompt or global SSL override. The original file is unchanged. Transfers use verified HTTPS, bounded retries, `.part` resume bound to remote validators, byte counts and local SHA-256 receipts. No server-published checksum is available from this downloader; local SHA-256 is a resume identity, not an official reference checksum.

Official train/validation lists are pinned to ScanNet commit `3830fce7f8b2e48ef047ef7fd76ea5f62903f51c`. The provisional acquisition plan follows the frozen physical-family hash ordering and excludes `scene0011`, `scene0050`, `scene0084`, `scene0168`, `scene0231`, `scene0378`, `scene0518`. Historical ScanNet200 five-scene inputs are contained in that exclusion set. The other `sceneNNNN` training aliases inspected in `configs/observation_query/splits*.json` refer to 3RScan UUIDs; they are not ScanNet identities.

| Role | Provisional captures |
|---|---|
| FIT | scene0547_00, scene0434_00, scene0639_00, scene0027_00, scene0107_00, scene0538_00, scene0571_00, scene0106_00 |
| CAL | scene0056_00, scene0534_00 |
| SELECT | scene0445_00, scene0626_00 |
| CONFIRM | scene0553_00, scene0064_00 |

All 14 `.sens` HEAD requests returned HTTP 200, totaling 9,871,549,087 bytes (9.19 GiB). Annotations and exported frames are additional. This check does not establish complete availability or 200-frame eligibility: after authorization, six file types and metadata frame counts must pass before writing `acquisition_lock.json`. Missing files permit the frozen same-family capture fallback; transient server failures abort instead of changing the selection. No teacher/model outcome influences acquisition.

The streaming sensor exporter preserves original JPEG payloads, uint16 millimeter depth, camera matrices and original frame IDs. The schedule is computed using the native negative-step rule and restricted to its first 200 slots. The mapper must consume the recorded positive `step` and explicit `end`, rather than recomputing the step from the sparse export. The official pinned native ScannetLoader loaded a synthetic 200-frame export with correct BGR registration, meter conversion and poses. This is format integration evidence only; real-data parity is pending data access. Confirmation export/mapping is deferred until frozen selection.

Existing native RGB/depth/pose exports can now satisfy input availability without requiring the `.sens` file again; missing scheduled frame IDs are recorded and never backfilled. Explicit dataset roots with official split files no longer need a particular basename.

Verification: 108 module tests pass, including real curl HTTPS Range resume against a trusted local test server, changed-content rejection, truncated-sensor rejection, physical-family fallback and outage handling. Scoped Ruff checks pass.

```bash
cd /home/ww/crove/ovimap-module-validation
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/prepare_ovimap_scannet.py \
  --data-root /mnt/shared/ww/ovimap-module-validation-v1/data/scannet --phase plan

# Execute only after the user confirms existing authorization and terms agreement:
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/prepare_ovimap_scannet.py \
  --data-root /mnt/shared/ww/ovimap-module-validation-v1/data/scannet \
  --phase download --authorized-scannet-download

/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/prepare_ovimap_scannet.py \
  --data-root /mnt/shared/ww/ovimap-module-validation-v1/data/scannet --phase export

OVIMAP_DATA_ROOTS=/mnt/shared/ww/ovimap-module-validation-v1/data/scannet \
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_module_study.py \
  --spec docs/paper/static_ovmap/module_validation_v1/spec/PROTOCOL_SPEC.json \
  --output-root /mnt/shared/ww/ovimap-module-validation-v1 --phase bind
```

Evidence: `artifacts/static_ovmap/module_validation_v1/scannet_preparation_20260922.json`. The acquisition plan is provisional, separate from the experiment's final `splits.json`.
