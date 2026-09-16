"""Explicit ONLINE setup only; runtime entry points never call this module."""

import argparse
import importlib
import json
from pathlib import Path

from local_runtime import sha256


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download",
        action="store_true",
        required=True,
        help="Explicitly authorize network asset acquisition for this invocation",
    )
    parser.add_argument("--root", default="workspaces/local-agent")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--only", choices=["all", "model", "traces", "eval"], default="all")
    args = parser.parse_args()
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    provenance = {}
    if args.only in ("all", "model"):
        repo = "google/gemma-4-E2B-it"
        revision = HfApi().model_info(repo, revision=args.model_revision).sha
        snapshot_download(
            repo,
            revision=revision,
            local_dir=root / "models/gemma-4-E2B-it",
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja", "*.txt"],
        )
        provenance["model"] = {"repo": repo, "revision": revision}
    if args.only in ("all", "traces"):
        repo = "badlogicgames/pi-mono"
        revision = HfApi().dataset_info(repo, revision=args.dataset_revision).sha
        snapshot_download(
            repo,
            repo_type="dataset",
            revision=revision,
            local_dir=root / "data/pi-mono",
            allow_patterns=["*.jsonl"],
        )
        provenance["traces"] = {"repo": repo, "revision": revision}
    if args.only in ("all", "eval"):
        dest = root / "data/eval"
        dest.mkdir(parents=True, exist_ok=True)
        h = importlib.import_module("inspect_evals.humaneval.humaneval")
        m = importlib.import_module("inspect_evals.mbpp.mbpp")
        write_rows(
            dest / "humaneval.jsonl",
            load_dataset(h.DATASET_PATH, revision=h.HUMANEVAL_DATASET_REVISION, split="test"),
        )
        write_rows(
            dest / "mbpp.jsonl",
            load_dataset(m.DATASET_PATH, "sanitized", revision=m.MBPP_DATASET_REVISION, split="test"),
        )
        write_rows(
            dest / "mbpp-prompt.jsonl",
            load_dataset(m.DATASET_PATH, "full", revision=m.MBPP_DATASET_REVISION, split="prompt"),
        )
        manifest = {
            "humaneval_revision": h.HUMANEVAL_DATASET_REVISION,
            "mbpp_revision": m.MBPP_DATASET_REVISION,
            "files": {p.name: sha256(p) for p in dest.glob("*.jsonl")},
        }
        (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
        provenance["eval"] = manifest
    (root / f"assets-{args.only}.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
