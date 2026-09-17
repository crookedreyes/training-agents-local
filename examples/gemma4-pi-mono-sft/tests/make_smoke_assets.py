"""Create random tiny model + synthetic traces locally; NEVER a quality benchmark."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_runtime import configure_local_runtime, sha256


def main():
    root = configure_local_runtime("workspaces/local-agent") / "fixtures"
    root.mkdir(exist_ok=True)
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    torch.manual_seed(42)
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[EOS]": 2,
        "user": 3,
        "assistant": 4,
        "hello": 5,
        "world": 6,
        "answer": 7,
        "tool": 8,
    }
    raw = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]"
    )
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + eos_token + ' ' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
    model_path = root / "model"
    tokenizer.save_pretrained(model_path)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(vocab),
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=2048,
            pad_token_id=0,
            eos_token_id=2,
        )
    )
    model.save_pretrained(model_path)
    traces = root / "traces"
    traces.mkdir(exist_ok=True)
    for index in range(4):
        events = [
            {"type": "session", "id": f"synthetic-session-{index}"},
            {"type": "message", "id": "u", "message": {"role": "user", "content": f"hello {index}"}},
            {"type": "message", "id": "a", "message": {"role": "assistant", "content": "world answer"}},
        ]
        (traces / f"{index}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    dest = root / "eval"
    dest.mkdir(exist_ok=True)
    row = {
        "task_id": "synthetic/0",
        "prompt": 'def answer():\n    """Return 42."""\n',
        "canonical_solution": "    return 42\n",
        "entry_point": "answer",
        "test": "def check(candidate):\n    assert candidate() == 42\n",
    }
    (dest / "humaneval.jsonl").write_text(json.dumps(row) + "\n")
    mbpp = {
        "task_id": 100,
        "prompt": "Write answer() returning 42.",
        "test_list": ["assert answer() == 42"],
        "source_file": "synthetic",
        "code": "def answer():\n    return 42",
        "test_imports": [],
    }
    (dest / "mbpp.jsonl").write_text(json.dumps(mbpp) + "\n")
    (dest / "mbpp-prompt.jsonl").write_text(
        "".join(
            json.dumps(
                {"task_id": i, "text": mbpp["prompt"], "test_list": mbpp["test_list"], "code": mbpp["code"]}
            )
            + "\n"
            for i in [2, 3, 4]
        )
    )
    (dest / "manifest.json").write_text(
        json.dumps({"synthetic_fixture": True, "files": {p.name: sha256(p) for p in dest.glob("*.jsonl")}})
    )
    print(root)


if __name__ == "__main__":
    main()
