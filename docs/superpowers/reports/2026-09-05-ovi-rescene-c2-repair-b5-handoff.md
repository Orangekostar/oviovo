# OVI-MAP x ReScene C2 Repair B5 Handoff

The frozen B5 attempt was stopped legally at C2. Do not run the backend against
the partial sidecar and do not fill missing rows with palette colors, synthetic
normals, cross-visit RGB-D, or zeros.

The highest-value next action is a new, separately versioned input study for
the 36,738 unsupported model rows. It should distinguish out-of-view geometry,
occlusion/depth-residual rejection, and source-normal gaps before changing any
policy. The current 0.05 m tolerance, visit windows, native sampler seed, model
checkpoint, and evaluated commit must remain immutable for this result.

No B4 topology was rebuilt because C2 failed first. The older B4 file is only
an aggregate summary and cannot support a topology comparison. GPU attempt,
B4 reconstruction, B5 comparison, and Office attempt counts are all zero.
