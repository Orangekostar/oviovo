# Final Apartment current-map views

The frozen `MV_QUALITY_GRAPH_BOUNDARY@H2` readout is exported as four PLY views
of the same 15,625,540 current source rows and 5,198,109 retained source triangles.
The original current-valid geometry and connectivity are preserved; no GT geometry,
smoothing, hole filling or generated image content is used.

RGB comes from the original `observed_rgb_uint8` field. The 13,382,630 valid RGB
rows retain their observed colors; invalid RGB is gray (128,128,128). Instance
colors are a stable hash of the unchanged numeric owner IDs. Semantic colors use
the same deterministic ID coloring and are recorded in the export manifest.
Colors do not change class or owner predictions.

The state panel is a display of the frozen visit/before/after membership:

| Color | Meaning | Source rows |
|---|---|---:|
| Green (30,160,75) | Current visit | 12,336,849 |
| Blue (45,105,190) | Retained historical visit | 3,285,752 |
| Amber (235,175,30) | Restored historical visit | 2,939 |

These display groups are not semantic evaluation roles or a new inference of
visibility/occlusion. In particular, amber does not assert correct semantics.

The PNG uses the existing renderer's fixed orthographic camera (-55° azimuth,
22° elevation) and up to 800,000 evenly spaced source indices for a tractable
overview. It is a display preview, not the full-resolution numeric evidence;
occluded geometry and small recovered regions may not be visible. The four
full PLY files retain every current row and explicit `canonical_source_row`,
owner, semantic, display-state and RGB-valid fields.

Full-map storage: `LOCAL_ONLY_POLICY`, approximately 2.08 GB total. Exact paths,
hashes, palette and input bindings are in `selected_apartment_map_exports.json`.
The complete source/connectivity audit is separate from preview rendering and
checks all rows rather than the displayed subset.
