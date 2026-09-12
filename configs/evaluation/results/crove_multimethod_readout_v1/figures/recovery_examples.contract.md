# Recovery diagnostic figure contract

Question: where do the frozen final readout's improvements and remaining errors
occur among all H2-restored source rows? Selection is the entire H2 minus B3
source set, not a GT-selected successful crop. Source geometry is unchanged.

Single X–Z projection, Python/matplotlib, 183 × 120 mm; editable SVG/PDF and
300-dpi PNG preview. Minimum text 7 pt. Plot all 2,939 source rows without jitter,
coordinate averaging, smoothing or ROI exclusion. Duplicate coordinates overlap;
legend counts must not be interpreted as independent samples or visible dot counts.

Categories use the existing source-to-GT diagnostic with strict 0.05 m nearest
support. They do not reproduce the headline GT-to-prediction mIoU projection.
Unknown support is gray, retained correctness blue, improvement green, regression
ochre and remaining error red. Zero regression is explicitly shown in the legend.
No uncertainty interval or significance claim is supported by this single window.

Audit source CSV hash/counts, inspect PNG and PDF text/collisions. Single-panel
alignment is not applicable; preserve the measured axes bounds. Prediction and
audit hashes are in the source manifest. No generated image content is used.
