# Research Notes

Use this file for durable lessons about Training Agents challenge design.

Suggested entry format:

```md
## YYYY-MM-DD - Short Title

- Challenge:
- Method:
- Model:
- Dataset:
- Reward/eval:
- What worked:
- What failed:
- Next:
```

## 2026-06-15 - Gemma 4 Pi-Mono SFT Sweep

- Challenge: SFT a small Gemma 4 instruction model on public coding-agent traces.
- Method: TRL `SFTTrainer`, prompt/completion conversion, completion-only loss,
  PEFT LoRA, 4-bit loading, hosted Trackio logging, and HF Jobs on `l40sx1`.
- Model: `google/gemma-4-E2B-it`.
- Dataset: `badlogicgames/pi-mono`, raw JSONL sessions converted locally inside
  the Job runner.
- Reward/eval: held-out eval loss and mean token accuracy on 256 converted
  examples; no task-completion eval yet.
- What worked: hosted Trackio Space creation with `space_id`, local script
  upload through `hf jobs uv run`, and Gemma 4 language decoder LoRA targets.
- What failed: broad LoRA suffixes targeted unsupported Gemma 4 wrapper modules;
  targeting `model.language_model.layers.*` projections fixed it.
- Next: add a task-level eval harness before treating the best adapter as an
  agent-quality improvement.

## 2026-09-10 - Local execution migration

- Method: TRL/PEFT completion-only NF4 LoRA; synthetic tiny Llama fixtures only.
- Runtime: local paths, local Trackio, no downloads/uploads at runtime; explicit
  setup is separate. Both RTX 5070 Ti GPUs passed independent offline smokes.
- Validation: two training steps, checkpoint resume to step four, adapter reload,
  generation, local Inspect fixture scoring and dashboard asset checks. Session
  groups and duplicate source files remain on one side of the validation split.
- Evidence and commands: `docs/local-validation.md`; artifacts/logs remain under
  ignored `workspaces/local-agent/runs/`.
- Limitation: no Gemma/Pi-Mono quality result or full benchmark claim. Real assets,
  full benchmark evaluation, Docker and distributed training remain unverified.
- Next: choose/import real assets, validate Gemma-specific memory and tokenizer
  behavior, then evaluate base and adapter with the frozen local protocol.

## 2026-09-11 UTC - Real Gemma 4 local run launched

- User requested the full actual experiment; real model, traces and benchmark
  data were acquired during explicit setup and pinned to revisions.
- Two benchmark-referencing sessions excluded; prepared split 5,687 / 265 rows,
  with session/duplicate grouping and recorded length/template filtering.
- Corrected Trainer's visible-GPU batch accounting; real single-GPU 4K smoke passed.
- Full 200-step NF4 LoRA run and base/adapter HumanEval + MBPP stages launched.
- Launch configuration: `docs/real-gemma4-run.md`. Live status, source snapshot,
  exact commands and outputs: `workspaces/local-agent/runs/real-gemma4/`.
- This is a launch record, not a completed training/evaluation result.

## 2026-09-11 UTC - Real Gemma4 local run completed

All stages of `gemma4-pi-mono-local-200` completed successfully: 200-step NF4 LoRA SFT and full base/adapter HumanEval and MBPP. SFT stage wall time was 72m21s; whole pipeline 3h40m27s. Validation loss at steps 50/100/150/200: 0.6562/0.6064/0.5709/0.5671; final token accuracy 85.50%. HumanEval pass@1: 117/164 -> 118/164 (15 gains, 14 regressions). MBPP pass@1: 66.85% -> 63.11%; pass@5: 73.54% -> 75.10%. All 2,898 samples scored with no sample-level evaluation errors. Mixed coding results do not establish an overall improvement or agent task-completion gains. Next state: assess held-out tool-use behavior on a development taskset before selecting another training variant. Detailed config, interpretation and evidence: `docs/real-gemma4-run.md`; local artifacts: `workspaces/local-agent/runs/real-gemma4/`.
