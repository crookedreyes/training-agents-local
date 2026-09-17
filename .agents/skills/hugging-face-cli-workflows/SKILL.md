---
name: hugging-face-cli-workflows
description: Use when working with Hugging Face CLI or Hub workflows for TRL training, including explicit setup downloads, offline asset bundles, revisions, dataset checks, and local model persistence.
---

# Hugging Face CLI Workflows

Use HF libraries locally. Network CLI/API actions belong only to an explicitly
requested asset-preparation step. Runtime never checks auth, downloads missing
assets, launches HF Jobs, uploads artifacts or creates Spaces.

- Prefer existing local model/data copies or an imported offline bundle.
- For requested acquisition, pin revisions and record hashes; never print tokens.
- Use `examples/gemma4-pi-mono-sft/prepare_assets.py` for explicit setup and
  `docs/local-execution.md` for local artifact layout and commands.
- Save adapters, tokenizer/template, run config, split manifest, metrics and logs
  under ignored local workspaces; adapters require their recorded local base model.

References: `references/hub-workflows.md` covers explicit acquisition/import;
`references/jobs-workflows.md` covers replacing remote jobs with local execution.
