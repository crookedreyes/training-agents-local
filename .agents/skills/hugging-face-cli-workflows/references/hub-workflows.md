# Asset acquisition and local artifacts

Use existing filesystem assets or import an offline bundle first. If the user
requests downloads, pin model/data revisions during setup and record hashes.
`prepare_assets.py --download` is separate from offline runtime commands.
Never call auth, Hub APIs or repository creation during training/evaluation.

Store base weights, tokenizer/template and shards together. Save adapters with
base-model hashes, run configuration, split manifest, metrics and eval logs under
ignored workspaces. An adapter alone cannot replace its base model.
See `docs/local-execution.md` for exact setup and runtime commands.
