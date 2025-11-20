# Repository Guidelines

## Project Structure & Module Organization
- Core library lives in `stereo/` (datasets, modeling, libs, utils).
- Experiment configs are in `cfgs/`; dataset metadata is under `data/`.
- Training, evaluation, and utility scripts are in `tools/` (for example, `tools/train.py`, `tools/eval.py`, `tools/infer.py`).
- Documentation and tutorials are under `docs/`; logs, checkpoints, and results should go to `output/`.

## Build, Test, and Development Commands
- Create the recommended environment:
  - `conda create -n openstereo python=3.8`
  - `pip install -r requirements.txt`
- Single‑GPU train example:  
  `python tools/train.py --cfg_file cfgs/lightstereo/lightstereo_s_sceneflow.yaml`
- Multi‑GPU DDP train example:  
  `torchrun --nnodes=1 --nproc_per_node=8 tools/train.py --dist_mode --cfg_file cfgs/lightstereo/lightstereo_s_sceneflow.yaml`
- Evaluation example:  
  `python tools/eval.py --cfg_file cfgs/lightstereo/lightstereo_s_sceneflow.yaml --eval_data_cfg_file cfgs/kitti15_eval.yaml --pretrained_model path/to/ckpt.pth`
- Inference example:  
  `python tools/infer.py --cfg_file cfgs/lightstereo/lightstereo_s_sceneflow.yaml --left_img_path left.png --right_img_path right.png`

## Coding Style & Naming Conventions
- Python only; use 4‑space indentation and keep line length reasonable.
- Prefer `snake_case` for functions/variables, `PascalCase` for classes, and config keys that match existing YAML patterns.
- Follow existing module boundaries (`stereo.datasets`, `stereo.modeling`, `stereo.utils`) and reuse helpers from `stereo/utils/common_utils.py` when possible.
- Avoid introducing new dependencies unless strictly necessary; if required, add them to `requirements.txt` and `docs/`.

## Testing Guidelines
- There is no formal unit‑test suite; rely on dataset‑specific evaluation.
- For functional validation, run `tools/eval.py` on at least one standard config (for example, KITTI or SceneFlow) and, where relevant, `tools/test_kitti.py` for KITTI submission outputs.
- When adding a new model or dataset, provide a minimal config under `cfgs/` and verify that training, evaluation, and inference all run without errors.

## Commit & Pull Request Guidelines
- Use clear, descriptive commit messages; prefix with a scope when helpful (for example, `feat(rlightstereo): ...`, `fix(datasets): ...`).
- Each PR should include: a short problem/solution summary, affected configs/datasets, example commands used to validate, and any benchmark numbers or qualitative results (tables, screenshots, or links).
- Update relevant docs in `docs/` and example configs in `cfgs/` when changing public behavior or adding new features.
- Keep changes focused and incremental; avoid mixing refactors, style cleanups, and new features in a single PR.

## Configuration, Data & Security
- Do not commit dataset files, proprietary weights, or credentials; reference local paths in configs instead.
- Keep YAML configs readable: comment non‑obvious options and align with existing naming schemes.
- For deployment or TensorRT changes, mirror patterns in `deploy/` and document any extra system requirements.

