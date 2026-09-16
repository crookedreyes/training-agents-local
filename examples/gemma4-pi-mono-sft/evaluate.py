"""Pinned Inspect coding tasks using local assets, inference and isolated scoring."""

import argparse
import importlib
import json
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from local_runtime import (
    adapter_manifest,
    configure_local_runtime,
    file_manifest,
    load_local_model,
    local_directory,
    sha256,
)
from sandbox import check_sandbox, execute


def read_rows(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Prepare benchmark input before evaluation: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected nonempty JSONL records: {path}")
    ids = [str(row["task_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate task IDs in {path}")
    return rows


@contextmanager
def local_task(benchmark: str, data_dir: Path, backend: str, image: str):
    """Keep upstream prompts/extraction/scorers; substitute only data and transport.

    Patches are scoped to one serial evaluation. Never call from concurrent tasks.
    """
    from datasets import Dataset
    from inspect_ai.dataset import MemoryDataset
    from inspect_ai.util import ExecResult

    module = importlib.import_module(f"inspect_evals.{benchmark}.{benchmark}")
    rows = read_rows(data_dir / f"{benchmark}.jsonl")

    def dataset_loader(*, sample_fields, **kwargs):
        return MemoryDataset(
            [sample_fields(row) for row in rows],
            name=f"local-{benchmark}",
            location=str(data_dir / f"{benchmark}.jsonl"),
        )

    class LocalSandbox:
        async def exec(self, cmd, timeout=30, **kwargs):
            if cmd[:2] != ["python", "-c"] or len(cmd) != 3:
                raise ValueError("Unexpected benchmark verifier command")
            result = execute(cmd[2], backend, timeout, image)
            if result.timed_out:
                raise TimeoutError("Verification timed out")
            return ExecResult(result.success, result.returncode, result.output, result.output)

    with (
        patch.object(module, "hf_dataset", dataset_loader),
        patch.object(module, "sandbox", lambda: LocalSandbox()),
    ):
        if benchmark == "mbpp":
            prompt_rows = read_rows(data_dir / "mbpp-prompt.jsonl")
            if not {2, 3, 4}.issubset({row["task_id"] for row in prompt_rows}):
                raise ValueError("MBPP requires the original few-shot prompt tasks 2, 3, 4")
            with patch.object(module, "load_dataset", lambda *a, **kw: Dataset.from_list(prompt_rows)):
                task = module.mbpp()
        else:
            task = module.humaneval()
        task.sandbox = None  # The original scorer now calls the explicit isolated local executor.
        yield task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--benchmark", choices=["humaneval", "mbpp"], required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--runtime-dir", default="workspaces/local-agent")
    parser.add_argument("--sandbox", choices=["bubblewrap", "docker"], default="bubblewrap")
    parser.add_argument("--sandbox-image", default="local-agent-eval:1")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.max_tokens <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("Token budget and optional sample limit must be positive")
    configure_local_runtime(args.runtime_dir)
    data_dir = local_directory(args.data_dir, "Benchmark")
    check_sandbox(args.sandbox, args.sandbox_image)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new evaluation output directory")
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Benchmark manifest.json is required; run explicit asset preparation first")
    manifest = json.loads(manifest_path.read_text())
    required = [f"{args.benchmark}.jsonl"] + (["mbpp-prompt.jsonl"] if args.benchmark == "mbpp" else [])
    for name in required:
        if manifest.get("files", {}).get(name) != sha256(data_dir / name):
            raise ValueError(f"Benchmark checksum mismatch: {name}")
    import torch
    from inspect_ai import eval
    from inspect_ai.model import GenerateConfig, Model, ModelAPI, ModelOutput, modelapi
    from transformers import set_seed

    set_seed(args.seed)
    model, tokenizer = load_local_model(
        args.model_path, args.adapter_path, args.device, args.load_in_4bit, args.bf16
    )
    model.eval()
    model.config.use_cache = True

    @modelapi(name="local-agent")
    class LocalAPI(ModelAPI):
        generated_count = 0

        async def generate(self, input, tools, tool_choice, config):
            if tools:
                raise ValueError("Coding evaluation expects text generation, not model-side tools")
            messages = [{"role": message.role, "content": message.text} for message in input]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt").to(model.device)
            temperature = config.temperature or 0
            kwargs = {
                "do_sample": temperature > 0,
                "max_new_tokens": config.max_tokens or args.max_tokens,
                "pad_token_id": tokenizer.pad_token_id,
            }
            if temperature > 0:
                kwargs["temperature"] = temperature
            started = time.monotonic()
            with torch.inference_mode():
                generated = model.generate(**inputs, **kwargs)
            content = tokenizer.decode(generated[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
            self.generated_count += 1
            print(
                f"phase=generation count={self.generated_count} input_tokens={inputs.input_ids.shape[1]} "
                f"output_tokens={generated.shape[1] - inputs.input_ids.shape[1]} seconds={time.monotonic() - started:.2f}",
                flush=True,
            )
            return ModelOutput.from_content(model=self.model_name, content=content)

    output.mkdir(parents=True, exist_ok=True)
    config = GenerateConfig(
        temperature=0 if args.benchmark == "humaneval" else 0.5, max_tokens=args.max_tokens, seed=args.seed
    )
    record = {
        **vars(args),
        "protocol": "inspect-local-v1",
        "smoke_only": args.limit is not None,
        "benchmark_manifest": manifest,
        "base_model_files": file_manifest(Path(args.model_path)),
        "adapter_files": adapter_manifest(Path(args.adapter_path)) if args.adapter_path else None,
    }
    (output / "protocol.json").write_text(json.dumps(record, indent=2))
    with local_task(args.benchmark, data_dir, args.sandbox, args.sandbox_image) as task:
        logs = eval(
            task,
            model=Model(LocalAPI("local-checkpoint"), config),
            log_dir=str(output),
            log_format="json",
            limit=args.limit,
            max_samples=1,
            max_connections=1,
            fail_on_error=True,
            epochs=1 if args.benchmark == "humaneval" else None,
            temperature=config.temperature,
            max_tokens=args.max_tokens,
        )
    summary = [
        {"status": log.status, "results": log.results.model_dump(mode="json") if log.results else None}
        for log in logs
    ]
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if any(log.status != "success" for log in logs):
        raise RuntimeError("Evaluation did not complete; inspect local logs")


if __name__ == "__main__":
    main()
