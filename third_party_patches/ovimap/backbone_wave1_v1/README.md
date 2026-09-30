# Backbone wave-1 native layer

Pinned upstream: OVI-MAP/OVI-MAP at
`f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424`.

Apply `../module_validation_v1/ovimap_module_validation_v1.patch` first, then
`backbone_wave1_v1.patch`. This layer contains the native-v10 factor-zero
raycast ownership export correction, opt-in deferred metadata/Python hooks,
read-only prior-depth probe, existing ratio switch, object-plan enforcement
through candidate recomputation/mode4 counts/alias compatibility, and explicit
realized-owner diagnostics. Existing settings are unchanged when disabled.

The runner builds an isolated extension at the protocol's native_build_root.
Its receipt records actual consumed source, base/new patch and loaded binary
identities. The historical native-query-v10 extension is never overwritten.

`python -m static_ovmap.backbone_wave1.native_patch` regenerates the layered
patch and verifies that applying both layers reproduces the scoped upstream
working files exactly, using a temporary Git index.
