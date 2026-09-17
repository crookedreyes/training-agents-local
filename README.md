# Training Agents

Public Codex context for agentic post-training work with TRL.

This repository contains reusable instructions, sub-agent definitions, skills,
and lightweight guides for planning, implementing, reviewing, and monitoring
agent training workflows.

It is not a training codebase. Keep checkpoints, datasets, logs, and experiment
outputs outside the tracked repo, usually under ignored `workspaces/`
directories or separate project repositories.

## Local execution

Training, inference, evaluation, tracking and artifacts run on this workstation.
Hugging Face libraries remain supported; cloud services are not runtime dependencies.
Start with [the local setup and validation guide](docs/local-execution.md).

## Examples

- `examples/gemma4-pi-mono-sft/`: local TRL/PEFT SFT, explicit asset preparation,
  offline inference, local Trackio, and isolated Inspect coding evaluations.
- GRPO, environment RL and self-distillation remain staged guides, not implemented
  training pipelines. Historical remote records live under `docs/` and `research/`.

## Guides

- `program.md`: operating model for Training Agents.
- `docs/program.md`: staged challenge ladder from SFT to environment GRPO and
  self-distillation.
- `docs/looping-rl.md`: blog post on loop-shaped reinforcement learning for
  agent training systems.
- `docs/terminal-bench-loop.md`: loop-shaped automation contract for training
  an approximately 2B open model toward Terminal-Bench performance above 40.
