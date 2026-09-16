# Real Gemma 4 local experiment

The user requested a full real run after the synthetic migration checks. The
pipeline completed on 2026-09-11 UTC on `feat/local-agent-training`. All six
stages exited successfully. Detailed logs and artifacts remain under ignored
workspaces.

## Completed results

SFT took 72 minutes 21 seconds including preparation and final validation;
Trainer recorded 68 minutes 36 seconds. The complete pipeline, including base
and adapter coding evaluations, took 3 hours 40 minutes 27 seconds.

| Benchmark | Base | SFT adapter | Change (percentage points) |
| --- | ---: | ---: | ---: |
| HumanEval pass@1 | 71.34% (117/164) | 71.95% (118/164) | +0.61 |
| MBPP pass@1, estimated from five samples/task | 66.85% | 63.11% | -3.74 |
| MBPP pass@2 | 70.00% | 69.14% | -0.86 |
| MBPP pass@5 | 73.54% (189/257) | 75.10% (193/257) | +1.56 |

All 2,898 generated samples were scored, with zero sample-level evaluation
errors. Base and adapter protocols differ only in adapter files/path and output
directory. Paired HumanEval outcomes show 15 newly solved tasks and 14 lost
tasks. MBPP had 859/1,285 passing base samples versus 811/1,285 adapter samples:
the adapter covered slightly more tasks across five attempts, but individual
attempts passed less often.

Validation loss decreased from 0.6562 at step 50 to 0.6064 at step 100, 0.5709
at step 150, and 0.5671 at step 200. Final validation token accuracy was 85.50%.
These measure imitation of held-out Pi-Mono completions, not successful task
execution. No step-zero validation baseline was collected. Training completed
200 steps (0.563 epochs), with average training loss 0.6397 and peak GPU
allocated memory 8.41 GiB.

The local execution succeeded, but this run does not demonstrate a consistent
coding improvement. HumanEval's net gain is only one task, while MBPP pass@1
declined. This is one training seed and one evaluation configuration; no
significance or general agent capability claim is established. The next useful
check is a held-out tool-calling/task-completion evaluation aligned with the
Pi-Mono training objective, followed by training changes selected using a
development set rather than repeated tuning on these benchmark tests.

Evidence: `adapter/trainer_state.json`, `adapter/train_results.json`, the four
evaluation `summary.json` files and sample logs, `result-analysis.json`, and
`*.status.json`, all under `workspaces/local-agent/runs/real-gemma4/`.

## Configuration

- Model: `google/gemma-4-E2B-it`, revision
  `3e22461f65e89153144f8adb70e3b8c2cc9845a7`.
- Source data: `badlogicgames/pi-mono`, revision
  `dac2a1d3ba12dda597b973a791a77618ccb5f413`.
- Review: 627 sessions downloaded; two with benchmark references excluded.
  The automated scan is not a complete contamination/privacy audit.
- Prepared split: 5,687 training examples / 265 validation examples. Complete
  session/duplicate groups stay on one side of the split. Length/template filters
  and exclusions are recorded in the conversion report.
- Method: completion-only TRL SFT, NF4 LoRA rank 16 / alpha 32, LR 2e-4,
  max length 4096, batch 1, accumulation 16, 200 optimizer steps, seed 42.
- Evaluation/save interval: every 50 steps, plus final saved-adapter validation.
- Coding evaluation: all 164 HumanEval test tasks, greedy pass@1; all 257 MBPP
  sanitized test tasks with five samples each at temperature 0.5. Base and adapter
  use identical input versions, prompts, precision and 768-token output budgets.
- GPU: explicit `cuda:0`; runtime training has OS network isolation. Evaluation
  uses Python network guards and network-isolated local verifier subprocesses.
- Tracking: local Trackio, `http://127.0.0.1:7860`, project `training-agents-sft`,
  run `gemma4-pi-mono-local-200`. Do not print token-bearing dashboard startup logs.

The initial real-model check exposed Trainer counting both visible GPUs when
computing batch size. The implementation now explicitly sets the pinned Trainer's
single-GPU accounting and checks actual batch size. The corrected real smoke
completed two steps with 4K filtering, about 8.23 GiB peak allocated memory, and
one example per microbatch. Earlier synthetic tests established execution and
resume, but did not catch that accounting issue.

## Commands and artifacts

Launch command used:

```bash
workspaces/local-agent/.venv/bin/python -u \
  workspaces/local-agent/runs/real-gemma4/pipeline.py \
  > workspaces/local-agent/runs/real-gemma4/pipeline.log 2>&1
```

Do not launch a second copy while the current run is active. The pipeline stops
on any failed stage. Each stage records its exact expanded command, PID, times,
exit status and output in `<stage>.status.json` and `<stage>.log`.

Under `workspaces/local-agent/runs/real-gemma4/`:

- `pipeline.status.json`: live stage, PID and completed/failed status.
- `train-200.log`: training progress and periodic validation.
- `adapter/`: run config, split/input manifests, checkpoints and final adapter.
- `base-humaneval/`, `base-mbpp/`, `adapter-humaneval/`, `adapter-mbpp/`:
  evaluation protocols, Inspect logs and summaries, created as stages run.
- `source/`: source snapshot and hashes used by the pipeline.
- `trace-audit.json`, `plan.json`: exclusions, dataset/model provenance and run plan.
- `final-run-report.md`: generated by the report watcher when the pipeline either
  completes or fails. Its absence means no final report has been produced yet.

Early optimizer timing was approximately 20 seconds/step, before periodic
validation. Historical turn-shuffled validation results are not comparable to
this grouped split; coding scores are not Terminal-Bench scores.
