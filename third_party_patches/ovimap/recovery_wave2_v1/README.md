# Recovery Wave 2 Native Layer

Pinned OVI-MAP upstream: `f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424`.
Use an isolated checkout and apply these layers in order:

```bash
git apply /path/to/oviovo/third_party_patches/ovimap/module_validation_v1/ovimap_module_validation_v1.patch
git apply /path/to/oviovo/third_party_patches/ovimap/backbone_wave1_v1/backbone_wave1_v1.patch
git apply /path/to/oviovo/third_party_patches/ovimap/recovery_wave2_v1/recovery_wave2_v1.patch
```

The layer adds opt-in per-group `USE_NATIVE` and `ASSIGN_EXISTING` behavior
through candidate construction/recomputation, native fresh-label allocation,
mode4 instance counts, explicit-owner reservation and transitive aliases.
Unmatched groups retain the complete native fallback path without Python
preallocation. Multiple accepted groups may follow one prior owner while
remaining separate current superpoints. Mixed fallback constraints are logged.

It also exposes the read-only depth-valid prior probe and the completed-frame
mode4 assignment snapshot. The latter is preserved before native temporary
state is cleared; factor0 accumulated count ownership is reported separately.
Disabled/off behavior matches the original native extension in the actual
serial-three-frame TSDF, labels, owners, aliases and raster validation.

From the oviovo repository, with the supplied environments and original build
dependencies available, reconstruct the isolated binary using the measured
original compiler/link commands:

```bash
PYTHONPATH=src:. /home/ww/miniconda3/envs/ovimap-map/bin/python \
  -m static_ovmap.recovery_wave2.runtime \
  --binding /mnt/shared/ww/ovimap-recovery-wave2-v1/attempt_001/resolved_inputs.json \
  --upstream /mnt/shared/ww/ovimap-recovery-wave2-v1/upstream \
  --build-root /mnt/shared/ww/ovimap-recovery-wave2-v1/tooling/recovery_native_v2 \
  --resume
```

The runtime exports and verifies the recovery patch against the preceding two
layers using a temporary Git index. The receipt binds compiler commands,
effective C++ sources, patch stack and the loaded binary. It does not overwrite
the original parent extension. The actual verified binary SHA is
`48cfc80599c7ec7849749d18eff163680e6c1d937b9e1aa37d73b764f963af1b`;
another compiler/runtime can produce different bytes and requires its own
native off/fallback/follower validation before full maps.

The initial diagnostic build and its failed follower getter traces are retained
externally. Historical consumed Python wrappers are archived with byte-exact
checksums in the compact validation evidence, including the early FC wrapper
that preceded the committed GPU-self-occupancy and input-hash guard fixes.
They are provenance artifacts; the current runner executes the final code.
