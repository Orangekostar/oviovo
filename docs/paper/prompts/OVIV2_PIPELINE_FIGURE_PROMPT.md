# OVIV2 Pipeline Figure Prompt

## Paper Pipeline

OVIV2 is an online RGB-D semantic mapping method whose central design is a
dual readout that separates historical scene evidence from the current scene
state:

```text
RGB-D frame + pose/intrinsics
        + frame-local object observations + dense semantic probabilities
                                |
                                v
                   Shared causal frame processing
                                |
               +----------------+----------------+
               |                                 |
               v                                 v
  Cumulative readout                    Temporal current readout
  Historical object ownership           1. local tracking + association
  + dense semantic fusion               2. active/dormant re-ID
  Persistent scene evidence             3. gated ICP, translation fallback
                                        4. object-submap integration
                                        5. signed visibility evidence
                                        6. probabilistic lifecycle:
                                           active / uncertain / dormant
                                        7. protected-object masking +
                                           temporal background integration
               |                                 |
               +----------------+----------------+
                                |
                                v
      Causal checkpoint t: cumulative map + current-state snapshot
                                |
                                v
        current objects/background, temporal IDs, lifecycle metadata
```

The cumulative branch retains historical scene evidence. The temporal branch
exports only objects judged to be currently present and the background
integrated after masking protected objects. Every checkpoint consumes data only
through its current frame and never reads future frames.

Keep development evaluation outside the method runtime as a separate evidence
strip:

```text
A0-A4 causal ablations -> immutable candidate evidence
-> common-v2 / occlusion / official metrics -> frozen selection
-> Apartment + Office formal repeats
```

## Drawing Prompt

```text
Create a publication-quality full-width vector pipeline figure for an AAAI-style
robotics / 3D vision paper. White background, flat scientific diagram, crisp
editable-looking vector text, no photorealism, no gradients, no decorative
icons, no numerical results.

Title inside the figure: "OVIV2 Dual-Readout Online Semantic Mapping".

Use a left-to-right causal data flow with five aligned stages.

Stage 1, Input Stream:
Show a compact RGB image tile, depth map tile, camera-pose coordinate frame,
and two narrow side inputs labeled "Frame-local Object Observations" and
"Dense Semantic Probabilities". Merge them into one box labeled
"Causal Frame t".

Stage 2, Dual Readout:
Split the causal frame into two parallel horizontal lanes.

Upper lane, muted gray-blue, label "Cumulative Readout".
Show "Persistent Object Ownership" and "Dense Semantic Fusion", producing
"Historical Scene Evidence". Make this lane visually stable and conservative.

Lower lane, primary OVIV2 blue/teal, label "Temporal Current Readout".
Draw four sequential modules:
(1) "Local Tracking & Association" with active and dormant entity tokens,
(2) "Gated Re-ID + ICP" with a small dashed fallback arrow labeled
"translation fallback",
(3) "Object Submap Update" with small local voxel blocks and a semantic
prototype token,
(4) "Visibility-backed Lifecycle" showing a compact state strip
ACTIVE -> UNCERTAIN -> DORMANT.
Use signed-depth / occlusion rays from the RGB-D input into this lifecycle
module, labeled "present / visible-absent evidence".

Stage 3, Background Handling:
From active and uncertain object submaps, draw a protection mask over the
depth image, then route unmasked evidence into a green module labeled
"Masked Temporal Background". This module and the active object submaps merge
into "Current-State Map".

Stage 4, Causal Output:
Show a checkpoint symbol labeled "Causal Checkpoint t".
It emits two clearly separated outputs:
"Cumulative Map (history)" and
"Current Snapshot (present objects + recovered background)".
Under the current snapshot, show three small metadata chips:
"temporal ID", "lifecycle", "causal timestamp".
Add a small lock symbol and label "no future frames".

Stage 5, Development Evidence Chain:
Place a thin light-gray dashed strip underneath the main pipeline, visually
separate from the method. Label it "Development evidence, not method runtime".
Show:
"A0-A4 ablations" -> "causal metrics + occlusion stress test + official
evaluation" -> "frozen selection" -> "Apartment / Office repeated runs".
Do not show scores, rankings, or claims of improvement.

Use a restrained, color-blind-safe scientific palette:
OVIV2 temporal branch #0072B2,
background branch #009E73,
lifecycle / visibility evidence #E69F00,
causal checkpoint / metadata #CC79A7,
cumulative branch #666666,
neutral fills #F2F2F2.
Ensure all distinctions also use lane position, border style, and arrows rather
than color alone. Keep labels at final-paper readability, with generous spacing,
consistent rounded rectangles only for modules, and minimal arrows.
```

## Manuscript Integration

Use the figure as a full-width `figure*` in the Method Overview, referenced
before the float appears. Keep all in-figure text at least 9 pt at final paper
size, export as editable PDF/SVG, and verify grayscale readability.

Suggested caption:

> Overview of OVIV2. Each causal RGB-D frame updates a persistent cumulative
> readout and a lifecycle-aware temporal readout in parallel; the latter
> combines local object submaps, visibility-backed lifecycle inference, and
> object-masked background integration to export a current-state snapshot
> without future-frame access.

## Evidence Boundary

The figure must not contain unverified T2 values, SOTA labels, rankings, or
claims of improvement. A0-A4 is development evidence rather than part of the
runtime architecture.
