# Local migration validation

Branch: `feat/local-agent-training`. Baseline commit:
`97d3c23793367993b0f3255fe3368417c549b080`.
Validation used the locked Python 3.13 environment on the inspected dual RTX 5070 Ti
workstation. Run instructions are in [local execution](local-execution.md).

## What passed

- Nine regression tests: grouped splits, duplicate/related sessions, no-eval and
  insufficient-group handling, malformed JSON, tool-result coherence, missing
  shards, external network/remote tracking rejection, invalid CLI arguments and
  adapter manifest scope (some assertions share tests).
- Three opt-in host integration tests: isolated verifier success/failure/timeout,
  blocked networking/hidden home, and correct fixtures through the original
  Inspect HumanEval/MBPP scorers. MBPP retained five samples and pass@1/2/5 reducers.
- Actual NF4 GPU operations and LoRA training on each GPU separately, using a
  randomly initialized tiny Llama fixture and four synthetic Pi-Mono sessions.
  Each logged final smoke trains two steps, resumes checkpoint 2 to step 4, then
  reloads the saved adapter and generates locally.
- SFT, resume and generation completed with external network access disabled at
  the OS level by an outer Bubblewrap network namespace. Nonzero gradient norms
  and finite losses were logged; optimizer/scheduler/RNG checkpoint files exist.
- Local Inspect evaluation with actual fixture-model inference completed for one
  synthetic HumanEval task and five generations of one synthetic MBPP task.
  Generated code was scored inside network-isolated local Bubblewrap sandboxes.
- Local Trackio dashboard HTML and packaged JavaScript/CSS loaded inside a network
  namespace with no external connectivity. The test server was then stopped.
- Ruff checks, syntax parsing of all Python files, skill validation, agent TOML
  parsing and `git diff --check` passed.

These are infrastructure tests, not Gemma training results or coding benchmark
scores. The random model's fixture completions are not evidence of model quality.
Historical remote scores are preserved separately and have not been reverified.

## Reproduce and inspect

Create fixtures with `tests/make_smoke_assets.py`, then run from the repo root:

```bash
bwrap --unshare-net --dev-bind / / \
  workspaces/local-agent/.venv/bin/python \
  examples/gemma4-pi-mono-sft/tests/run_smoke.py \
  --device cuda:0 --output-dir workspaces/local-agent/runs/validated-synthetic-gpu0

bwrap --unshare-net --dev-bind / / \
  workspaces/local-agent/.venv/bin/python \
  examples/gemma4-pi-mono-sft/tests/run_smoke.py \
  --device cuda:1 --output-dir workspaces/local-agent/runs/validated-synthetic-gpu1

LOCAL_AGENT_SANDBOX_TEST=1 workspaces/local-agent/.venv/bin/pytest -q \
  examples/gemma4-pi-mono-sft/tests \
  --basetemp=workspaces/local-agent/runs/final-regression-tests

bwrap --unshare-net --dev-bind / / \
  workspaces/local-agent/.venv/bin/python \
  examples/gemma4-pi-mono-sft/tests/check_dashboard.py
```

Use **new** output directories when repeating training; the named directories
already contain the recorded runs. Each GPU smoke directory contains per-step
`*-command.json`, `*-output.log`, prepared traces and `adapter/` with run config,
input hashes, split manifest, local adapter, final eval metrics and checkpoints.
Trackio data is under `workspaces/local-agent/trackio/`.

Actual inference/eval smoke reports:

- `workspaces/local-agent/runs/smoke-gpu0-v2/eval-v2/`: HumanEval protocol,
  Inspect JSON log and summary; one synthetic sample completed.
- `workspaces/local-agent/runs/validated-synthetic-gpu1/mbpp-smoke/`: MBPP protocol,
  Inspect JSON log and summary; five synthetic samples completed.
- `workspaces/local-agent/runs/dashboard-check/`: dashboard command, log and local
  bundled-asset check result.

All environments, generated models, traces, checkpoints and logs remain ignored.

## Remaining checks and scope

- The real Gemma E2B weights, Pi-Mono traces and official benchmark snapshots were
  not downloaded during this implementation. Their local source/download choice
  is pending. Validate real-model memory, tokenizer/tool schema and training with
  those assets before a longer run; compare base and adapter on the fixed protocol.
- Docker was not installed or tested. Bubblewrap is the tested verifier backend.
  Docker requires a separately prepared local Python image and daemon access.
- This host restricts nested Bubblewrap namespaces. Complete evaluation was tested
  from the host with Python network guards and isolated verifier subprocesses;
  the entire evaluation process was not enclosed in the outer network namespace.
  The Python guard does not enforce policy on arbitrary native network clients.
- Distributed training is rejected. Both GPUs work independently; no DDP/FSDP
  profile or combined-memory behavior is claimed.
- GRPO, environment RL, preference training and self-distillation remain guides.
  Full Terminal-Bench needs its own executable harness, local assets, dependency
  audit and evaluation protocol.

The initial smoke exposed a removed `warmup_ratio` argument in the resolved
Transformers stack; the implementation uses `warmup_steps=0.03`. Local Trackio
uses an explicit trainer callback to avoid repeated initialization by the default
integration. Sandbox process limits account for existing Torch threads on the
host. These fixes are incorporated in the final workflow and tests.
