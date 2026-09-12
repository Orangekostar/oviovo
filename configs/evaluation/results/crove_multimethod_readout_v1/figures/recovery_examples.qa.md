# Recovery diagnostic QA

All 2,939 frozen restored source rows are included. Source CSV count/hash and
category counts are checked by the plotter. Both native and selected readout
correct/incorrect source and physical counts match the existing recovery audit.

Visual inspection: all categories and zero-regression legend entry are visible;
axes use equal metric scale. Duplicate coordinates overlap as declared. White
space preserves the actual spatial aspect ratio, not an omitted region.
This is a diagnostic projection of real source points, not an RGB crop or GT mesh.

Rendered PDF: minimum 7 pt across 35 text runs; no text below 5 pt. Collision
audit: zero failures and zero warnings. Single-panel alignment is not applicable;
the measured axes bounds and source hash are saved in the alignment JSON.
SVG/PDF are vector exports; PNG at 300 dpi is the preview. No uncertainty or
statistical significance is inferred from the display.
