# SAM3 API Probe - 2026-06-04

## Environment

- Worktree: `/home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates`
- Current env probe command: `/home/ww/miniconda3/envs/oviovo/bin/python scripts/probe_sam3_env.py --json`
- Current Python: `3.10.20 (main, Mar 11 2026, 17:46:40) [GCC 14.3.0]`
- Current torch import: `/home/ww/miniconda3/envs/oviovo/lib/python3.10/site-packages/torch/__init__.py`
- Current torch version: `2.6.0+cu124`
- Current CUDA availability: `true`
- Current CUDA version reported by torch: `12.4`
- Current SAM3 import: failed with `ModuleNotFoundError: No module named 'sam3'`
- Dedicated SAM3 env check: `/home/ww/miniconda3/envs/sam3` does not exist.
- Alternate env search: no runnable external Python with an importable `sam3` package was found under `/home/ww/anaconda3/envs`, `/home/ww/miniconda3/envs`, or project virtualenvs.
- SAM3 env creation attempt: `/home/ww/miniconda3/bin/conda create -y -p /tmp/sam3_env_probe python=3.12 --dry-run` failed because Anaconda channel Terms of Service have not been accepted in this noninteractive environment. No environment was created.

Official README prerequisites observed during the controller probe:

- Python 3.12 or higher.
- PyTorch 2.7 or higher.
- CUDA-compatible GPU with CUDA 12.6 or higher.
- Recommended install shape: create `sam3` conda env with Python 3.12, install `torch==2.10.0 torchvision` from CUDA 12.8 wheels, clone `facebookresearch/sam3`, then `pip install -e .`.
- Checkpoints require Hugging Face access/authentication to `facebook/sam3` or `facebook/sam3.1`.

Repo/source probe method:

- Official GitHub repository `facebookresearch/sam3` README and source were fetched during the controller probe.
- Local verification used `scripts/probe_sam3_env.py --json` in the existing `oviovo` env.
- No multi-GB torch/checkpoint download was attempted because the current env mismatches official Python/PyTorch/CUDA prerequisites and the dedicated SAM3 env is absent.
- `sam3.sam3_image_predictor` probe result: absent in the current official source tree. The tree contains example notebook `examples/sam3_image_predictor_example.ipynb`, but no importable `sam3/sam3_image_predictor.py` module; the image API is exposed through `sam3.model_builder` and `sam3.model.sam3_image_processor`.

## Confirmed API

Official image API from README/source:

- Builder import: `from sam3.model_builder import build_sam3_image_model`
- Builder signature observed: `build_sam3_image_model(bpe_path=None, device="cuda" if torch.cuda.is_available() else "cpu", eval_mode=True, checkpoint_path=None, load_from_HF=True, enable_segmentation=True, enable_inst_interactivity=False, compile=False)`
- Processor import: `from sam3.model.sam3_image_processor import Sam3Processor`
- Processor signature observed: `Sam3Processor(model, resolution=1008, device="cuda", confidence_threshold=0.5)`
- Image binding: `state = processor.set_image(PIL.Image)`
- Text prompt binding: `output = processor.set_text_prompt(state=state, prompt="<class prompt>")`
- Output fields: `output["masks"]`, `output["boxes"]`, `output["scores"]`
- Output semantics from source: masks are boolean thresholded tensors at original image size, boxes are `xyxy` in original image coordinates, scores are confidence values after thresholding.
- Legacy/alternate predictor module: `sam3.sam3_image_predictor` was checked and is not present in the current official source tree.

## Worker Mapping

- Backend request image format: JSON object with `shape: [H, W, 3]`, `dtype: "uint8"`, `encoding: "base64"`, and contiguous RGB byte data.
- Worker image conversion: base64 decode to `np.uint8` array, validate `[H, W, 3]`, convert to `PIL.Image` in `RGB` mode.
- Prompt format: each class string from `classes` is passed directly as one text prompt to `processor.set_text_prompt`.
- Frame processing: load SAM3 once at worker startup, call `processor.set_image(image)` once per request, then run all class prompts for that frame.
- Proposal mapping: one SAM3 mask becomes one proposal with `label` set to the class prompt, `confidence` from SAM3 score, `mask` as nested boolean lists, optional `bbox_xyxy` from SAM3 boxes, and `metadata.sam3_score` plus `metadata.prompt`.
- Worker response filtering: count raw masks before confidence filtering, filter by request `confidence_threshold`, sort proposals by descending confidence, cap to request `max_proposals`.
- Error behavior: load and inference failures return `{"id": <matching integer id>, "ok": false, "error": "<explicit message>"}`. There is no placeholder success fallback.

## Risks And Blockers

- Installation blocker: current `oviovo` env is Python 3.10.20, torch 2.6.0+cu124, CUDA 12.4, and has no `sam3` package; official prerequisites require Python >=3.12, PyTorch >=2.7, and CUDA >=12.6.
- Dedicated env blocker: `/home/ww/miniconda3/envs/sam3` is absent, and conda refused a noninteractive dry-run create until Anaconda channel Terms of Service are accepted.
- Checkpoint/auth blocker: builder default uses Hugging Face when no `checkpoint_path` is supplied; access/authentication to `facebook/sam3` or `facebook/sam3.1` is required.
- GPU memory risk: SAM3 image model load/inference may exceed available GPU memory; this was not measurable without a runnable SAM3 env and checkpoint.
- Prompt batching risk: official confirmed API is one text prompt call at a time after `set_image`; batching multiple class prompts in one processor call was not confirmed, so the worker loops over classes.
- Runtime compatibility risk: using the current `oviovo` env would fail before model load due to missing package and version mismatch; real runs should use a dedicated SAM3 Python executable via worker config.
