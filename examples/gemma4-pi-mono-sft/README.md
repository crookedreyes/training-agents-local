# Local Gemma 4 Pi-Mono SFT template

Train with local Transformers/TRL/PEFT, generate with local weights, score code
with local Inspect and Bubblewrap or Docker, and view local Trackio metrics.
Runtime commands require prepared filesystem assets and never download or upload.

See [setup, commands and validation](../../docs/local-execution.md).
The previous cloud runs are preserved in the
[historical record](../../docs/historical-gemma4-remote-run.md); their scores have
not been reproduced by this migration.

Files:

- `train_sft.py`: Pi-Mono conversion, grouped validation splits, completion-only
  LoRA training, local checkpoints, final evaluation and resume.
- `local_runtime.py`: offline policy, asset/device checks and model loading.
- `preflight.py`: tokenizer/assets and actual CUDA/quantization kernel smoke.
- `generate.py`: local base/adapter inference.
- `evaluate.py`: pinned Inspect HumanEval/MBPP tasks with local datasets and
  inference. Keeps upstream prompts, extraction, tests and pass@k aggregation.
- `sandbox.py`: network-isolated local code verification with time/resource limits.
- `dashboard.py`: loopback Trackio UI, without sharing/syncing.
- `prepare_assets.py`: separate, explicit **online setup**, never called at runtime.
- `pyproject.toml`, `uv.lock`: reproducible Python 3.13 environment.
- `tests/`: offline policy, split and asset regression tests; synthetic smoke fixtures.

This is a small reusable template, not a general training framework. Start with
one GPU. DDP is explicitly rejected until a distributed profile is validated.
Gemma-specific LoRA targets remain the default; another architecture requires
explicit targets. Artifacts and environments belong in ignored `workspaces/`.
