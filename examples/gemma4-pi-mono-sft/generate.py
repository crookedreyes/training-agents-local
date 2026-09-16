"""Generate with a local base checkpoint and optional local LoRA adapter."""

import argparse
import json

from local_runtime import configure_local_runtime, generate_text, load_local_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runtime-dir", default="workspaces/local-agent")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.temperature < 0:
        parser.error("Token budget must be positive and temperature nonnegative")
    configure_local_runtime(args.runtime_dir)
    from transformers import set_seed

    set_seed(args.seed)
    model, tokenizer = load_local_model(
        args.model_path, args.adapter_path, args.device, args.load_in_4bit, args.bf16
    )
    print(
        json.dumps(
            {
                "model_path": args.model_path,
                "adapter_path": args.adapter_path,
                "seed": args.seed,
                "completion": generate_text(
                    model, tokenizer, args.prompt, args.max_new_tokens, args.temperature
                ),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
