"""Validate prepared assets and exercise this machine's CUDA/quantization kernels."""

import argparse
import importlib.metadata
import json
import shutil

from local_runtime import configure_local_runtime, local_directory, validate_device, validate_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--runtime-dir", default="workspaces/local-agent")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-eval-sandbox", action="store_true")
    args = parser.parse_args()
    root = configure_local_runtime(args.runtime_dir)
    model = validate_model(args.model_path)
    raw = local_directory(args.raw_dir, "Raw trace")
    files = list(raw.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No session JSONL in {raw}")
    validate_device(args.device, args.load_in_4bit)
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    tokenizer.apply_chat_template(
        [{"role": "user", "content": "Test."}], tokenize=False, add_generation_prompt=True
    )
    x = torch.randn(8, 64, device=args.device, requires_grad=True)
    if args.load_in_4bit:
        import bitsandbytes as bnb

        layer = bnb.nn.Linear4bit(64, 64, compute_dtype=torch.bfloat16, quant_type="nf4").to(args.device)
    else:
        layer = torch.nn.Linear(64, 64).to(args.device)
    loss = layer(x).square().mean()
    loss.backward()
    if not torch.isfinite(loss) or not torch.isfinite(x.grad).all():
        raise RuntimeError("Kernel smoke produced nonfinite values")
    if args.require_eval_sandbox:
        from evaluate import check_sandbox

        check_sandbox()
    print(
        json.dumps(
            {
                "status": "ok",
                "model": str(model),
                "raw_files": len(files),
                "device": args.device,
                "cuda_build": torch.version.cuda,
                "disk_free_bytes": shutil.disk_usage(root).free,
                "versions": {
                    n: importlib.metadata.version(n)
                    for n in ("torch", "transformers", "trl", "peft", "bitsandbytes", "trackio", "inspect-ai")
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
