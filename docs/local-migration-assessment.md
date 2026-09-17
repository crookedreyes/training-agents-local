# Local execution assessment and proposed migration

Prepared 2026-09-10 against commit `97d3c23793367993b0f3255fe3368417c549b080` on `main`.

Historical assessment, approved by the user before implementation on `feat/local-agent-training`. Statements about absent code/environment refer to the pre-migration snapshot. Current commands and limitations are in [local execution](local-execution.md).

**Decision and scope**

Keep PyTorch, Transformers, TRL, PEFT, Accelerate, datasets, bitsandbytes, and other libraries that execute locally. Move all experiment computation, inference, scoring, tracking, storage, and execution control onto this workstation. Runtime must succeed without internet access or cloud credentials and fail clearly when local assets are missing.

Allow a separate, explicit preparation process for obtaining packages, model files, datasets, and container images. Alternatively, import the same files from an offline bundle. Preparation is never an automatic fallback from a training or evaluation command. No download is authorized or performed by this assessment.

The implemented SFT workflow and its documented coding evaluations are the migration scope. Localize instructions for the other methods, but label their implementations as future work. The `.codex/agents/` files describe research-assistant roles; they are not trained models or an inference service implemented by this repository. This migration does not replace the coding assistant application itself.

**1. Repository inventory**

Reviewed the 49 tracked files, including the single Python implementation (872 lines), all six local skills and their references, eight agent definitions, operational guides, and research records. There is no project dependency lock, test suite, local inference entry point, checked-in benchmark runner, GRPO/DPO/RLOO implementation, environment server, or executable self-distillation loop. The README explicitly identifies this as a context repository with a lightweight example.

| Area | Evidence | Current behavior and migration |
| --- | --- | --- |
| Training | `examples/gemma4-pi-mono-sft/train_sft.py:646` | TRL SFTTrainer with LoRA, optional NF4, completion-only loss. Preserve the method; localize inputs, device placement, logging, saves and resume. |
| Trace conversion | Same file, lines 243–508 | Pi-Mono raw session JSONL becomes rendered prompt/completion rows. Preserve provenance and tool structure; improve validation and session splitting. |
| Training data | Same file, line 511 | `--raw-dir` already bypasses download; absent that option, `snapshot_download` fetches Hub data. Require an explicit local data path. |
| Model/tokenizer | Same file, lines 530 and 725 | `from_pretrained` accepts a remote model ID and has no local-only enforcement. Require local paths and local-only loading. |
| Tracking | Same file, lines 647–699 | Trackio supports local logging, but optional Spaces and inherited environment settings can route it remotely. Enforce local storage and local UI. |
| Artifacts | Same file, lines 702, 768 and 824 | Optional Hub repository creation and uploads coexist with local saving. Remove active upload paths and use a local artifact manifest. |
| Execution | Example README, lines 30–100 | Even “Local Smoke” selects a hosted Space; sweep uses HF Jobs. Replace with workstation commands. |
| Inference/eval | Example README, lines 136–228 | Remote HF Job launches vLLM/Inspect, installs packages at runtime, references Hub weights/data, and executes code. Add local entry points with prepared assets and isolated local execution. |
| Future training ladder | `docs/program.md` and skill references | SFT, GRPO, environment RL, distillation, and preference methods are guidance; only SFT is executable. Revise guidance to local inputs and execution. |
| Recurring automation | `docs/terminal-bench-loop.md:29`, `:97`; `docs/looping-rl.md` | Another user's absolute memory path, named Hopper cluster, SLURM, FSx, Spaces and publishing requirements. Replace operational defaults with local state, processes, GPUs and artifact promotion. |
| History | `research/notes.md`, `research/results.tsv`, example README | Prior remote results are reported but underlying logs/checkpoints are absent here. Preserve as historical records, not newly verified local results. |

There are no inference-provider API calls in the implemented training script. The migration mainly removes remote orchestration and asset/telemetry dependencies, and supplies missing local inference/evaluation entry points.

**2. Workstation assessment**

| Component | Observed |
| --- | --- |
| OS | Ubuntu 26.04.1 LTS, x86-64 |
| CPU | AMD Ryzen 9 9900X, 12 cores / 24 threads |
| RAM | About 29 GiB total, 25 GiB available when inspected; 8 GiB swap |
| GPU 0 and GPU 1 | NVIDIA RTX 5070 Ti; each 16,303 MiB total, 15,828 MiB free at inspection |
| NVIDIA driver | 595.84 |
| Disk | About 2.5 TiB available on the filesystem containing the repository |
| `/tmp` | RAM-backed tmpfs; do not use for model/data caches or training outputs |
| Python | Default 3.14.4; uv-managed Python 3.13 installations also present |
| ML environment | No repo `.venv`; default Python cannot import torch, transformers, TRL, PEFT, datasets, Trackio, Inspect or vLLM; no pip in that interpreter |
| Container runtime | Neither Docker nor Podman found on PATH |
| Assets | No experiment workspace. Checked HF cache lacks the required Gemma E2B/Pi-Mono assets; other model cache directories exist. This was not a whole-disk asset search. |

The initial sandbox could see PCI hardware but not GPU device files. A permitted read-only `nvidia-smi` outside that sandbox succeeded. The driver is functioning; future GPU runs need an execution context with GPU access.

Recommended starting configuration: the existing Gemma 4 E2B model, NF4 LoRA, one GPU, batch size 1, gradient accumulation, checkpointing, and 1,024-token smoke sequences before trying 2,048 and 4,096 tokens. Begin with 1–5 optimizer steps and measure peak VRAM and throughput. This is a feasibility hypothesis, not a measured training result. The model's effective parameter label is insufficient for budgeting embeddings, auxiliary modules, activations, optimizer state and cache memory.

Treat the GPUs as two separate 16 GB devices. DDP replicates the model and does not create a single 32 GB memory pool. Initially run training and evaluation sequentially. After single-GPU validation, test the same run on GPU 1 and offer an explicit two-process DDP profile if each replica fits. A separate local rollout/evaluation GPU can be considered later. Do not promise proportional speedup or enable sharding without measuring it.

Resolve and lock a compatible Python/PyTorch/CUDA/Transformers/TRL/PEFT/bitsandbytes combination. Python 3.13 is a candidate already installed, not a compatibility guarantee. Blackwell requires an appropriate PyTorch build and quantization kernels; the bitsandbytes installation matrix lists `sm120` in CUDA 12.8 and newer build families. Verify the selected version using real forward/backward and quantized-operation smoke tests. [bitsandbytes installation](https://huggingface.co/docs/bitsandbytes/installation), [PyTorch installation](https://pytorch.org/get-started/locally/).

**3. Correctness findings to address during migration**

1. **Session leakage:** `make_dataset_splits` shuffles assistant-turn examples, although several examples share a session and overlapping conversation history. A dependency-free reproduction using the actual function and a stub Dataset returned six sessions present in both splits for 200 synthetic turns from ten sessions, with seed 42 and ten eval rows. This proves the split mechanism can leak session content; actual historical overlap cannot be quantified without the data. Split by stable session/task group before expanding turns; group duplicates and related sessions when provenance supports it. Freeze manifests and keep final benchmarks separate.
2. **Uncontrolled network access:** local data alone does not make the script offline. Model/tokenizer loading, Trackio settings, uploads, package resolution and eval dataset loading remain separate network paths. Enforce local paths, configure offline behavior before imports, reject remote runtime destinations and validate with egress blocked. Transformers documents `HF_HUB_OFFLINE=1` and `local_files_only=True`. [Transformers offline operation](https://huggingface.co/docs/transformers/installation).
3. **Tracking environment precedence:** `setdefault` preserves inherited destinations; `space_id=None` does not neutralize an inherited `TRACKIO_SPACE_ID`. Validate/remove remote Space/server/storage/webhook settings in the launched process, establish the chosen local directory before importing Trackio, and test UI assets offline. Trackio documents these destination overrides. [Trackio API](https://huggingface.co/docs/trackio/api), [environment variables](https://huggingface.co/docs/trackio/environment_variables).
4. **Device placement:** `device_map="auto"` leaves placement implicit on this dual-GPU host. Use a selected single device, or explicit per-rank placement for a separately tested DDP path. Record world size and effective batch size; coordinate file writes and tracking on rank zero.
5. **Evaluation can be stale:** the saved eval metrics are the last periodic log entry, which may precede the final checkpoint. There is no unconditional final evaluation or best-checkpoint selection. Evaluate the final saved adapter and associate every score with a specific checkpoint hash; optionally promote the best validation checkpoint with a separate held-out gate.
6. **Recovery and portability:** `trainer.train()` has no resume argument exposed. Verify checkpoints contain the optimizer/scheduler/RNG state needed to resume, save tokenizer/chat template explicitly, and record the local base model relationship. An adapter alone is not a complete standalone model.
7. **Input/trace robustness:** negative `eval_size` is accepted; in the reproduction `-1` gives 199 eval rows and one train row. Unknown JSON shapes can fail after parsing. Context trimming can orphan tool results or clip tool arguments; broad rendering exceptions silently discard examples. Add explicit validation and rejection counts, preserve coherent tool-call/result groups, and verify completion masks with the actual tokenizer. `--target-modules` also loses precedence to the nonempty default regex despite the help suggesting otherwise.
8. **Reproducibility:** dependencies are open-ended minimum versions; the eval image is `latest` and installs packages at runtime. Pin resolved packages and container digests, record dataset/model revisions and hashes, and avoid changing benchmark prompt/scoring semantics accidentally. The historical example's metrics cannot be assumed comparable after fixing the split.

**4. Proposed implementation phases, after approval**

1. **Branch and local contract.** Recheck the worktree, create `feat/local-agent-training`, and record this approved design in `docs/local-execution.md`. Add small reusable helpers beside the existing example rather than building a general training platform. Keep environments, assets, runs and generated reports under ignored `workspaces/`.
2. **Environment and asset preparation.** Add example-scoped dependency configuration and a lock; separate training from optional accelerated inference dependencies if their requirements conflict. Add a preflight that checks paths, tokenizer/config/weight completeness, package versions, CUDA, quantized kernels, selected GPU memory, disk and eval runtime availability. Document importable offline bundles and an optional explicit preparation command. Include model weights, tokenizer/template/config, raw Pi-Mono JSONL, HumanEval test data, MBPP sanitized test data and its few-shot prompt data, and required container images. Missing assets produce an actionable error with no download attempt. Maintain checksums and provenance.
3. **Local SFT and validation fixes.** Refactor the existing script to require local model and raw-data inputs, enforce offline loading, remove upload/Space options, choose the GPU explicitly, and retain TRL/PEFT completion-only LoRA. Implement grouped splits, input validation, mask checks, local tracking, unique run directories, final eval, local save and resume. Preserve the current Gemma-specific decoder LoRA targeting; switching models would be a separately recorded experiment.
4. **Local inference and coding evaluation.** Add a minimal `generate.py` using local Transformers + PEFT loading, followed by a local `evaluate.py` that uses Inspect with locally supplied datasets and the same model/adapter. Prefer in-process inference initially; only add vLLM acceleration after compatibility and output-protocol validation. Inspect supports local model paths. Use small local task wrappers where built-in tasks otherwise fetch data, preserving pinned prompts, code extraction, tests and metrics. Execute generated code in prepared local Docker containers with network disabled, limits/timeouts and disposable directories. Do not expose the host workspace or credentials to generated code. [Inspect local model support](https://inspect.aisi.org.uk/providers.html), [sandboxing](https://inspect.aisi.org.uk/sandboxing.html).
5. **Local operating instructions.** Update `AGENTS.md`, both program guides, loop guides, relevant agent definitions, skill instructions/references, and example README. Default to local processes, local Trackio, workstation resource budgets, filesystem artifacts and repo-relative loop memory. Retain HF libraries and historically accurate remote research records. Replace active HF Jobs/SLURM/FSx/Space/upload recipes; mark remote CLI material as optional asset preparation/reference where retained. Keep explicit-user-request-only delegation behavior.
6. **Acceptance and handoff.** Run the targeted tests below, document exact successful commands and asset hashes, and provide local dashboard/artifact paths plus remaining limitations. First prove the full path on a small run, then run base-versus-adapter coding evaluation with the frozen protocol. A smoke pass establishes operability, not model improvement.

Suggested lightweight additions under `examples/gemma4-pi-mono-sft/`: `local_runtime.py`, `preflight.py`, `generate.py`, `evaluate.py`, narrowly scoped offline dataset/task helpers, local eval container configuration, dependency/lock files and focused tests. Exact module boundaries can remain small if code reuse permits. No large datasets, checkpoints, logs or full experiment output enters Git.

**5. Local evaluation protocol**

SFT training data remains Pi-Mono sessions converted to prompt/completion data; prompts, tool observations and padding must be masked from the target loss. Use a seed-controlled session-group train/validation split and persist membership. Enforce a clean no-eval mode when explicitly requested or reject insufficient groups rather than silently reverting to turn splitting.

Compare base and adapter on identical fixed HumanEval and MBPP inputs, prompts, budgets and generation settings. HumanEval full evaluation covers 164 test tasks with greedy pass@1; MBPP uses the chosen pinned sanitized test set, its separate prompt examples, temperature 0.5 and five samples per task for pass@1/2/5. Confirm counts against the imported manifests rather than assuming them. Store errors/timeouts and denominators, not just successful responses. A small subset is labeled smoke-only. The current upstream tasks load benchmark data by Hub ID, and MBPP also loads a separate few-shot split; both paths must be localized. [HumanEval source](https://raw.githubusercontent.com/UKGovernmentBEIS/inspect_evals/main/src/inspect_evals/humaneval/humaneval.py), [MBPP source](https://raw.githubusercontent.com/UKGovernmentBEIS/inspect_evals/main/src/inspect_evals/mbpp/mbpp.py).

Task verifiers run locally; neither benchmark needs a remote model judge. Also report tool-call schema validity and, where a small local terminal harness exists, held-out task success. Coding benchmark scores alone do not establish terminal-agent ability.

Full Terminal-Bench/Harbor execution is a later rung: no runnable harness exists here, and each selected task needs an audit for internet/package/service dependencies. Prepare images and task assets locally and label any altered/offline subset as a proxy. Do not promise the documented score-above-40 target from an infrastructure migration.

**6. Acceptance criteria and proposed commands**

The commands below describe the proposed interface; the new scripts/options do not exist yet and these commands have not been run. The virtual environment and assets must already be prepared.

```bash
workspaces/local-agent/.venv/bin/python examples/gemma4-pi-mono-sft/preflight.py \
  --model-path workspaces/local-agent/models/gemma-4-E2B-it \
  --raw-dir workspaces/local-agent/data/pi-mono --device cuda:0

workspaces/local-agent/.venv/bin/python examples/gemma4-pi-mono-sft/train_sft.py \
  --model-path workspaces/local-agent/models/gemma-4-E2B-it \
  --raw-dir workspaces/local-agent/data/pi-mono \
  --work-dir workspaces/local-agent/prepared/pi-mono \
  --output-dir workspaces/local-agent/runs/sft-smoke \
  --device cuda:0 --max-length 1024 --max-steps 2 \
  --per-device-train-batch-size 1 --gradient-accumulation-steps 1 \
  --logging-steps 1 --eval-steps 1 --save-steps 1 --seed 42

workspaces/local-agent/.venv/bin/python examples/gemma4-pi-mono-sft/generate.py \
  --model-path workspaces/local-agent/models/gemma-4-E2B-it \
  --adapter-path workspaces/local-agent/runs/sft-smoke \
  --device cuda:0 --prompt 'Write a Python function that reverses a string.'

workspaces/local-agent/.venv/bin/python examples/gemma4-pi-mono-sft/evaluate.py \
  --model-path workspaces/local-agent/models/gemma-4-E2B-it \
  --adapter-path workspaces/local-agent/runs/sft-smoke \
  --benchmark humaneval --data-dir workspaces/local-agent/data/eval \
  --output-dir workspaces/local-agent/runs/sft-smoke/eval \
  --device cuda:0 --sandbox docker --limit 8 --seed 42
```

Trackio data: `workspaces/local-agent/trackio/`; dashboard bound to loopback, with the exact launch command/URL documented for the pinned version. Artifacts: `workspaces/local-agent/runs/<run-id>/`, containing adapter, tokenizer/template, config, command, split manifest, version/hash manifest, checkpoints, metrics and evaluation logs. Base weights remain under `workspaces/local-agent/models/`.

Required acceptance checks:

- Preparation, SFT, save/reload, inference, evaluation and dashboard work with external network access denied and credentials absent; capture attempted egress as well as successful connections. Containers prohibit external traffic; loopback is available only where local components require it.
- Missing model shards, tokenizer files, raw data, benchmark inputs or images fail before expensive computation with no fallback fetch.
- Remote upload/Space/server destinations and inherited remote tracking settings cannot bypass the local policy. Dependency installation is not performed at runtime.
- Splits have no session/group overlap; masks exclude prompts/tool results/padding; malformed traces, zero-eval and insufficient-data cases behave predictably.
- At least one real GPU SFT smoke completes with finite loss and nonzero trainable-adapter gradients, saves locally, reloads and generates. Exercise save/resume and final-checkpoint evaluation.
- Evaluation verifier fixtures include a correct answer, an incorrect answer and a timeout. Base and adapter use identical settings and produce separate auditable reports.
- GPU 1 can run the single-GPU path. Enable the optional two-GPU training profile only after a dedicated DDP smoke proves per-rank placement, effective batch accounting and coordinated writes.

**7. Work beyond the implemented workflow**

GRPO can later use TRL with local prompt-only datasets, local generation and deterministic rewards; start with tiny groups and measure reward variance/memory. DPO/RLOO/reward modeling need their own datasets and tested trainer configurations. Environment RL requires a local reset/step/state harness and versioned reward functions. Distillation should use locally generated, verified traces and optional local teacher/judge models, with held-out acceptance gates. These are new implementations, not code presently waiting to be converted. Localize their guides now and schedule concrete experiments after the SFT/eval foundation passes.

**8. Validation performed for this assessment**

Read-only repository inventory and dependency/network-path search; AST syntax parse of the only Python script; successful `python3 examples/gemma4-pi-mono-sft/train_sft.py --help`; extraction and execution of the existing split function with synthetic inputs and a Dataset stub; CPU/RAM/disk/OS/interpreter checks; default-interpreter module availability and bounded asset-cache checks; successful host `nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv`; official library/evaluator documentation and source review; clean `git status --short --branch` on `main`.

No real training, tokenization, quantization, inference, benchmark evaluation, CUDA tensor operation, or library integration test has run. Those depend on the prepared environment and required assets. Exact memory needs, throughput, package compatibility and historical metric comparability remain unmeasured. Approval of this plan starts branch-based implementation; it does not make the historical scores newly verified or guarantee improved model quality.
