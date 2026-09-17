# Local SFT commands

The tested local entry point is `examples/gemma4-pi-mono-sft/train_sft.py`.
Use `docs/local-execution.md` for exact environment/preflight/train/resume commands.
It requires `--model-path` and `--raw-dir`; runtime never accepts Hub IDs as a
substitute for missing local assets. Use `--prepare-only` to validate conversion.

For new TRL templates, validate the installed CLI/config schema before providing
commands. Configure completion masking, explicit device, grouped validation,
local tracking and artifacts, and no Hub push or hosted Space destination.
