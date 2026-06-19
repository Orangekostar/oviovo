## Task: Build the backbone of a sustainable open-vocabulary dynamic mapping system

You are helping me build a research codebase from scratch.

Your job is to create the **project backbone only**, not a fake full implementation.

The target system is an **online open-vocabulary dynamic mapping framework** with:

* a global background map
* a set of object-centric local maps
* instance-first association
* object-level semantic memory
* lifecycle-aware dynamic maintenance

This stage is only for **main architecture construction**.

You must focus on:

* project structure
* module decomposition
* data structures
* interfaces
* pipeline wiring
* placeholder implementations
* config system
* logging
* simple runnable demo

Do **not** pretend to fully solve the algorithms.

---

## 1. Important reference codebases

You are allowed to inspect and borrow design ideas from the following open-source local repositories:

* `/home/phl/vv/paper2/OVO`
* `/home/phl/vv/paper2/DualMap`

Use them as references for:

* code organization
* pipeline design
* object-centric mapping logic
* semantic update logic
* configuration style
* data structures
* utility organization

Do **not** blindly copy everything.
Extract only useful backbone patterns.

In addition:

* OVO can be referenced for online open-vocabulary mapping pipeline design, segment/object-centric semantic aggregation, and general module organization.
* DualMap can be referenced for object-centric map maintenance, observation structuring, and dynamic scene adaptation logic.
* OVI is **not open source**, so for OVI-inspired parts you must create reasonable abstractions manually.

---

## 2. High-level method constraints

### 2.1 Core map representation

The system has **two layers**:

#### Background map

* one global TSDF-like map
* stores stable static scene layout
* stores large-scale background geometry

#### Object maps

* a set of object instances
* each object maintains:

  * object id
  * pose
  * local geometry
  * 3D bbox
  * semantic memory
  * observation history
  * lifecycle state
  * confidence

---

### 2.2 Methodology constraints

Follow this philosophy:

1. build class-agnostic object instances first
2. associate new observations mainly by spatial/geometric consistency
3. semantics must **not** dominate instance association
4. semantics are updated only at the object level
5. background and objects must be maintained separately
6. support long-term ghost cleanup and re-identification

---

### 2.3 Reference methodology guidance

Use these ideas as guidance:

#### OVI-inspired philosophy

* instance-first
* spatial-consistency-first association
* semantics updated after object identity is stabilized

#### OVO-inspired philosophy

* online open-vocabulary mapping pipeline
* multi-view object semantic aggregation
* clean modular structure for mapping

#### DualMap-inspired philosophy

* object-centric online maintenance
* object observations as explicit units
* dynamic scene handling hooks

For OVI-inspired parts:

* do not assume code exists
* create clean placeholders and interfaces manually

---

## 3. What to build

Build a **minimal but rigorous backbone**.

I want around **8-10 major modules max**.

Each module must have:

* clear responsibility
* explicit input types
* explicit output types
* clean placeholder implementation
* extensible interface
* TODO markers for future algorithm filling

I also want:

* central pipeline runner
* config system
* minimal demo
* logging hooks
* unit-test stubs
* README for the backbone

---

## 4. Required modules

Please structure the system into the following modules.

### Module 1. Frame Input Module

Responsibility:

* load RGB-D frame
* load or receive camera pose
* package them into a standard frame object

Input:

* RGB image
* depth image
* pose

Output:

* `Frame`

---

### Module 2. Proposal Module

Responsibility:

* produce class-agnostic 2D object proposals
* no semantic labeling here

Input:

* `Frame.rgb`
* `Frame.depth`

Output:

* list of `Proposal2D`

Implementation note:

* create a swappable backend interface for SAM / FastSAM / similar
* placeholder implementation is enough for now

---

### Module 3. Depth Refinement Module

Responsibility:

* refine 2D masks using depth discontinuity / geometric boundary cues

Input:

* `Frame.depth`
* list of `Proposal2D`

Output:

* refined list of `Proposal2D`

Implementation note:

* placeholder geometric refinement is enough
* interface must be clear

---

### Module 4. Patch Lifting Module

Responsibility:

* lift refined 2D proposals into 3D object patches in world coordinates

Input:

* refined proposals
* depth
* pose
* camera intrinsics

Output:

* list of `Patch3D`

Each `Patch3D` should include:

* points
* centroid
* bbox
* timestamp
* optional normals

---

### Module 5. Background/Object Split Module

Responsibility:

* decide whether each 3D patch belongs to background, object, or ambiguous region

Input:

* list of `Patch3D`
* current `BackgroundMap`
* current object set

Output:

* background patches
* object patches
* ambiguous patches

Implementation note:

* simple placeholder rules are enough
* leave hooks for geometry-based / motion-based / support-based logic

---

### Module 6. Object Association Module

Responsibility:

* associate object patches to existing object instances
* or request creation of new objects

Input:

* object patches
* object set

Output:

* `AssociationResult`

Important:

* association must be geometry-first / spatial-first
* semantics can only be a weak auxiliary signal

Design a scoring interface for:

* centroid distance
* bbox overlap
* geometry overlap
* pose proximity

---

### Module 7. Object Update Module

Responsibility:

* update object local geometry
* update pose / bbox / history
* call semantic update hook after association

Input:

* association result
* object patches
* object set

Output:

* updated object set

Implementation note:

* use local point cloud as v1 object geometry
* leave object-local TSDF as future extension hook

---

### Module 8. Semantic Memory Module

Responsibility:

* maintain object-level semantic memory
* no dense per-point language features
* update semantics only after object association is completed

Input:

* object id
* crop metadata
* view metadata
* optional embedding input

Output:

* updated semantic memory
* top-k label hypotheses

Implementation note:

* create a generic backend interface for SigLIP / CLIP / other VLM backends
* placeholder backend is enough for now
* include feature bank and confidence history in the data structure

---

### Module 9. Background Update Module

Responsibility:

* update the global background map
* avoid contamination from foreground object regions

Input:

* background patches
* background map
* object occupancy hints

Output:

* updated background map

Implementation note:

* use a lightweight TSDF-like placeholder structure
* no need to implement full TSDF fusion yet
* but design the interface so it can be replaced later

---

### Module 10. Dynamic Maintenance Module

Responsibility:

* long-term cleanup and lifecycle maintenance

Sub-functions:

* ghost trimming
* boundary contamination repair
* background reclaim
* lifecycle transitions
* re-identification hook

Input:

* object set
* background map
* current observations

Output:

* cleaned/updated maps and states

Implementation note:

* only implement simple placeholders
* focus on interfaces and state machine design

---

## 5. Required data structures

Define clear typed dataclasses / structs / classes for at least:

* `CameraIntrinsics`
* `Frame`
* `Proposal2D`
* `Patch3D`
* `ObjectState` enum
* `SemanticMemory`
* `ObservationRecord`
* `ObjectMap`
* `BackgroundMap`
* `AssociationScore`
* `AssociationResult`
* `SystemState`

Requirements:

* Python type hints
* clean docstrings
* readable and modular

---

## 6. Required project structure

Please create a clean Python project structure like this:

```
project_root/
├── configs/
├── data/
├── src/
│   ├── core/
│   ├── modules/
│   ├── models/
│   ├── utils/
│   └── pipelines/
├── tests/
├── run_demo.py
└── README.md
```

Inside `src/modules/`, organize modules clearly.

---

## 7. Engineering requirements

### Language

* Python

### Style

* typed
* modular
* research-friendly
* readable
* minimal but extensible

### Must include

* docstrings
* type hints
* TODO markers
* basic logging
* config object or yaml loader
* pipeline runner executing all modules in order
* minimal runnable demo with fake data

### Must avoid

* monolithic scripts
* pretending the real algorithms are already done
* dataset-specific hardcoding
* over-engineering

---

## 8. Required outputs

I want you to generate:

1. proposed folder structure
2. module breakdown
3. core data structure definitions
4. main pipeline code
5. placeholder implementations for each module
6. minimal runnable demo
7. README explaining:

   * modules
   * inputs/outputs
   * future extension points

---

## 9. Important design rules

1. Separate **algorithm skeleton** from **backend models**

   * proposal backend must be swappable
   * semantic backend must be swappable

2. Preserve the distinction between:

   * background layer
   * object layer

3. Preserve the distinction between:

   * instance association
   * semantic update

4. Make it easy to later upgrade:

   * local point cloud -> local TSDF
   * placeholder semantic backend -> SigLIP / CLIP
   * simple association -> stronger spatial voting / learned matching

5. For OVI-inspired parts:

   * define clear abstractions manually
   * add TODO notes
   * do not fake unavailable details

6. Reuse useful engineering patterns from:

   * `/home/phl/vv/paper2/OVO`
   * `/home/phl/vv/paper2/DualMap`

But keep the new codebase clean and independent.

---

## 10. Additional interface document

After building the backbone, also produce a concise interface document.

For each module, provide:

* module name
* responsibility
* input types
* output types
* internal state used
* extension points
* dependency on other modules

Also provide one system-level table:

| Step | Module | Input | Output | Notes |

Keep this part concise and implementation-oriented.

---

## 11. Execution note

Before writing code:

1. inspect `/home/phl/vv/paper2/OVO`
2. inspect `/home/phl/vv/paper2/DualMap`
3. summarize what is reusable for:

   * project structure
   * data structures
   * pipeline layout
   * object-centric maintenance
4. then implement the new backbone in a clean independent codebase

Do not tightly couple the new project to those repositories.
Use them only as references.

---

## 12. Final instruction

Do not overbuild.

Start from a **minimal but rigorous backbone** that can run end-to-end on one fake frame and print intermediate outputs.

The purpose is to let me inspect the architecture first, then iteratively fill in the real algorithms.
