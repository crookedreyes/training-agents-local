# Local execution

See [the recorded validation](local-validation.md) for checks actually run and
remaining limitations. The [approved assessment](local-migration-assessment.md)
records the original scope.

The implemented workflow is Gemma 4 E2B Pi-Mono supervised fine-tuning with
Transformers, TRL, PEFT/LoRA and optional NF4 quantization. All runtime computation,
inference, evaluation, tracking and artifact storage runs on this workstation.
The GRPO/environment/distillation ladder remains guidance for future experiments.

## Workstation and environment

Inspected machine: Ryzen 9 9900X, about 29 GiB RAM, two RTX 5070 Ti GPUs with
16,303 MiB each, driver 595.84, Ubuntu 26.04.1, and about 2.5 TiB free disk.
GPU memory is separate; do not treat the cards as one 32 GB device. Start with
`cuda:0`, batch size 1, checkpointing, NF4 and 1,024 tokens. Measure before raising
sequence length or concurrency. The template rejects multi-process/DDP launches.

GPU access and sandbox creation require host device/namespace access. The coding
agent's own sandbox may hide devices or prevent nested namespaces; a failed
in-sandbox `nvidia-smi` does not establish a broken driver.

Run commands from the repository root. Environment setup is a deliberate online
step; it installs the exact versions in the checked-in lock:

```bash
UV_PROJECT_ENVIRONMENT="$PWD/workspaces/local-agent/.venv" uv sync \
  --project examples/gemma4-pi-mono-sft --locked \
  --cache-dir workspaces/local-agent/cache/uv --python 3.13
```

The tested lock resolves PyTorch 2.14.0 + CUDA 13.0, Transformers 5.17.0,
TRL 1.13.0, PEFT 0.20.0, bitsandbytes 0.50.2, Trackio 0.37.1 and Inspect AI
0.3.263 / Inspect Evals 0.19.0. Use the lock, not open-ended upgrades. A newer
stack may require code and protocol changes. The machine already has Python 3.13
through uv; an offline machine also needs a compatible Python installation.

For offline installation, transfer the matching uv cache and Python installation
from a prepared machine, then use the same command with `--offline`. The cache,
lock, OS/architecture and Python version must agree. A copied `.venv` alone is not
necessarily relocatable. No package installation occurs in runtime entry points.

## Assets

Use an existing complete local Transformers checkpoint or explicitly acquire the
original model/data once. Runtime never falls back to an asset download. An adapter
requires the original base weights; GGUF inference files alone are not equivalent
to the training checkpoint. Copy the model config, tokenizer, chat template, weight
indices and all referenced shards together.

Optional online preparation, after choosing to acquire assets:

```bash
workspaces/local-agent/.venv/bin/python examples/gemma4-pi-mono-sft/prepare_assets.py \
  --download --only all
```

This resolves model/data revisions to commits, obtains the Gemma 4 E2B checkpoint,
raw Pi-Mono sessions, HumanEval test rows, MBPP sanitized test rows and MBPP's
separate few-shot prompt rows, and records their provenance. Use `--only model`,
`--only traces`, or `--only eval` for a partial setup. Authentication, if the source
requires it, is handled by HF's setup environment; runtime needs no credentials.

Alternatively import these directories from an offline bundle:

```text
workspaces/local-agent/
  models/gemma-4-E2B-it/       complete base checkpoint/tokenizer
  data/pi-mono/               one raw session per *.jsonl
  data/eval/                 humaneval.jsonl, mbpp.jsonl, mbpp-prompt.jsonl,
                             manifest.json with source revisions and SHA-256s
  prepared/                 derived JSONL and conversion reports
  runs/<run-id>/             adapter, checkpoints, manifests, metrics, eval logs
  trackio/                  local metrics database
  cache/                    local package/HF caches
  tmp/                      disk-backed temporary data
```

Do not store model caches on this host's RAM-backed `/tmp`. Use fresh output
folders for new runs; existing output requires an explicit compatible resume.
Review/redact traces before training. No user trace folders are read automatically.

## Training and inference

```bash
workspaces/local-agent/.venv/bin/python examples/gemma4-pi-mono-sft/preflight.py \
  --model-path workspaces/local-agent/models/gemma-4-E2B-it \
  --raw-dir workspaces/local-agent/data/pi-mono --device cuda:0

workspaces/local-agent/.venv/bin/python examples/gemma4-pi-mono-sft/train_sft.py \
  --model-path workspaces/local-agent/models/gemma-4-E2B-it \
  --raw-dir workspaces/local-agent/data/pi-mono \
  --output-dir workspaces/local-agent/runs/gemma-smoke \
  --device cuda:0 --max-length 1024 --max-steps 2 \
  --gradient-accumulation-steps 1 --logging-steps 1 --eval-steps 1 --save-steps 1 \
  --run-name gemma-smoke --seed 42

workspaces/local-agent/.venv/bin/python examples/gemma4-pi-mono-sft/generate.py \
  --model-path workspaces/local-agent/models/gemma-4-E2B-it \
  --adapter-path workspaces/local-agent/runs/gemma-smoke \
  --device cuda:0 --prompt 'Write a Python function that reverses a string.'
```

`--prepare-only` converts and validates local traces without training. The tokenizer
is still needed. Trace sessions are grouped before selecting train/validation rows;
byte-identical copies stay together. Supply `--split-groups groups.json` to group
related task/session filenames. Validation aims at `min(eval-size, 5% of rows)`
but keeps entire groups, so the actual row count can exceed the target. At least
two independent groups are required unless `--eval-size 0` is explicit. Record
`split_manifest.json`; never compare these splits directly to the historical
turn-shuffled validation scores.

Use the same command/config/output plus
`--resume-from-checkpoint workspaces/local-agent/runs/gemma-smoke/checkpoint-2`
and a larger `--max-steps` to continue a run. Configuration and split membership
must match. Checkpoints include trainer/optimizer/scheduler state. Final metrics
are evaluated after saving the final adapter; `artifact_manifest.json` links it
to hashed base assets. Model-quality comparisons must evaluate base and adapter
with identical prompts, token budgets, precision, tasksets and random seeds.

For an offline integration fixture with no pretrained downloads:

```bash
workspaces/local-agent/.venv/bin/python examples/gemma4-pi-mono-sft/tests/make_smoke_assets.py
```

It creates a random tiny Llama model and synthetic traces under `fixtures/`.
Use `fixtures/model`, `fixtures/traces`, `--target-modules q_proj,v_proj`,
`--max-length 128`, and a unique output/run name for SFT. These fixtures test
mechanics only and are never evidence of Gemma or agent-quality improvement.

## Local evaluation and isolation

Bubblewrap is installed and provides the default verifier on this host. It runs
Python with a fresh filesystem view, no home/workspace mounts, no network, a
read-only system runtime, temporary storage, bounded output and time/resource
limits. Docker is optional; prepare a Python image named `local-agent-eval:1`
locally, then use `--sandbox docker`. Docker runs use `--pull=never`, no network,
no capabilities, a non-root user and memory/process limits. No image is pulled by
evaluation. The template never runs generated code directly in the host workspace.

```bash
workspaces/local-agent/.venv/bin/python examples/gemma4-pi-mono-sft/evaluate.py \
  --model-path workspaces/local-agent/models/gemma-4-E2B-it \
  --adapter-path workspaces/local-agent/runs/gemma-smoke \
  --benchmark humaneval --data-dir workspaces/local-agent/data/eval \
  --output-dir workspaces/local-agent/runs/gemma-smoke/humaneval-smoke \
  --sandbox bubblewrap --device cuda:0 --limit 8 --seed 42
```

Remove `--adapter-path` for the base model and use a separate output directory.
Remove `--limit` for the full imported taskset. HumanEval uses greedy pass@1 and
one generation/task; MBPP uses temperature 0.5, five generations/task and upstream
pass@1/2/5 aggregation. The pinned upstream prompts, few-shot examples, code
extraction and verifiers are retained through scoped local data/transport adapters.
Do not invoke multiple concurrent evaluations in the same Python process.
`protocol.json`, Inspect logs and `summary.json` record settings and failures.
An 8-task smoke or synthetic fixture is not a benchmark score.

Python runtime enforces HF offline mode, local paths, disabled telemetry, and a
network audit guard. It rejects external DNS/connections and inherited remote
Trackio destinations; local loopback remains available. This guard covers Python
network calls, not arbitrary native libraries or subprocesses. To verify training
or inference under OS-enforced network denial, prefix the command with:

```bash
bwrap --unshare-net --dev-bind / / <command and arguments>
```

This prefix isolates networking only; it is **not** the generated-code filesystem
sandbox. This host restricts nested Bubblewrap namespaces, so run verifier checks
and evaluation directly from the host context; each generated-code verifier is
itself network-isolated. Do not describe the complete evaluation process as OS
network-isolated unless that separate configuration has actually been tested.

## Tracking, validation and next rungs

```bash
workspaces/local-agent/.venv/bin/python examples/gemma4-pi-mono-sft/dashboard.py \
  --project training-agents-sft --port 7860

workspaces/local-agent/.venv/bin/pytest -q examples/gemma4-pi-mono-sft/tests
```

The dashboard binds `http://127.0.0.1:7860`; data remains in
`workspaces/local-agent/trackio/`. Sharing, hosted Spaces, uploads, remote servers,
webhooks and remote themes are excluded. Inspect logs also remain on disk.

For loop automation use `workspaces/local-agent/loop/memory.md`, creating it on
first use. Preserve research history and record local commands, versions, hashes,
validation gates and the next state. GRPO uses local prompt tasks/generation and
verifiers; environment RL needs an implemented local reset/step/state harness;
distillation uses verified local traces and local teachers if needed. Full
Terminal-Bench requires a separate harness and task-dependency audit. Offline
subsets/proxies must be labeled; no score-above-40 claim follows from this migration.
