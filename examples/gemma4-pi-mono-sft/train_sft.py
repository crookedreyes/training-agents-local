"""SFT Gemma 4 E2B-it on badlogicgames/pi-mono coding-agent traces.

Offline template: reads prepared local raw session JSONL, converts visible
assistant/tool-call turns to prompt/completion examples, and uses TRL
completion-only loss so user prompts and tool outputs are not training targets.
"""

from __future__ import annotations

import argparse
import atexit
import copy
import hashlib
import importlib.metadata
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_runtime import (
    configure_local_runtime,
    file_manifest,
    local_directory,
    validate_device,
    validate_model,
)

DEFAULT_PROJECT = "training-agents-sft"
DEFAULT_RUN_NAME = ""
DEFAULT_LORA_TARGET_REGEX = (
    r".*language_model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
    r"$"
)


KNOWN_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "bash": {
        "description": "Run a shell command in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run."},
                "cmd": {"type": "string", "description": "Shell command to run."},
                "timeout": {"type": "number", "description": "Optional timeout in milliseconds."},
            },
            "required": [],
        },
    },
    "read": {
        "description": "Read a file or image from the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to read."},
                "file": {"type": "string", "description": "Path to read."},
                "offset": {"type": "number", "description": "Optional starting line."},
                "limit": {"type": "number", "description": "Optional line limit."},
                "start": {"type": "number", "description": "Optional starting line."},
                "end": {"type": "number", "description": "Optional ending line."},
            },
            "required": [],
        },
    },
    "edit": {
        "description": "Edit a file in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to edit."},
                "oldText": {"type": "string", "description": "Text to replace."},
                "newText": {"type": "string", "description": "Replacement text."},
                "edits": {"type": "array", "description": "Structured edits."},
                "patch": {"type": "string", "description": "Patch content."},
            },
            "required": [],
        },
    },
    "write": {
        "description": "Write content to a file in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to write."},
                "content": {"type": "string", "description": "File content."},
            },
            "required": [],
        },
    },
    "grep": {
        "description": "Search text in files.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Search pattern."},
                "path": {"type": "string", "description": "Path to search."},
                "limit": {"type": "number", "description": "Optional result limit."},
                "literal": {"type": "boolean", "description": "Treat pattern literally."},
                "context": {"type": "number", "description": "Context lines."},
            },
            "required": [],
        },
    },
    "find": {
        "description": "Find files or text in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Pattern to find."},
                "path": {"type": "string", "description": "Path to search."},
                "limit": {"type": "number", "description": "Optional result limit."},
            },
            "required": [],
        },
    },
    "ls": {
        "description": "List files in a directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path."},
                "limit": {"type": "number", "description": "Optional result limit."},
            },
            "required": [],
        },
    },
    "todo": {
        "description": "Manage a lightweight task list.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "Task-list action."},
                "text": {"type": "string", "description": "Task text."},
                "id": {"type": "string", "description": "Task identifier."},
            },
            "required": [],
        },
    },
}


@dataclass
class ConversionStats:
    files: int = 0
    events: int = 0
    message_events: int = 0
    user_messages: int = 0
    assistant_messages: int = 0
    tool_messages: int = 0
    skipped_messages: int = 0
    json_errors: int = 0
    image_parts: int = 0
    thinking_parts: int = 0
    assistant_examples: int = 0
    skipped_render_errors: int = 0
    skipped_prefix_mismatch: int = 0
    skipped_empty_completion: int = 0
    invalid_events: int = 0
    skipped_token_prefix: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--work-dir", default="workspaces/local-agent/prepared/pi-mono")
    parser.add_argument("--output-dir", default="workspaces/local-agent/runs/sft")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-tokenizer-check", action="store_true")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--eval-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-context-messages", type=int, default=18)
    parser.add_argument("--max-tool-result-chars", type=int, default=12000)
    parser.add_argument("--max-user-chars", type=int, default=12000)
    parser.add_argument("--max-assistant-chars", type=int, default=12000)
    parser.add_argument("--max-prompt-chars", type=int, default=64000)
    parser.add_argument("--include-reasoning", action="store_true")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--filter-overlength", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="",
        help="Comma-separated module suffixes. Leave empty to use --target-modules-regex.",
    )
    parser.add_argument("--target-modules-regex", default=DEFAULT_LORA_TARGET_REGEX)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trackio-project", default=DEFAULT_PROJECT)
    parser.add_argument("--trackio-group", default="pi-mono-sft-sweep")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--runtime-dir", default="workspaces/local-agent")
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument(
        "--split-groups", help="Optional JSON mapping source filenames to related task/session group IDs"
    )
    args = parser.parse_args()
    args.run_name = args.run_name or Path(args.output_dir).name
    for name in ("eval_size", "max_files", "max_examples"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative")
    for name in (
        "max_length",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "logging_steps",
        "eval_steps",
        "save_steps",
        "lora_r",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.device == "cpu" and (args.bf16 or args.load_in_4bit):
        parser.error("CPU smoke requires --no-bf16 --no-load-in-4bit")
    return args


def clip_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    omitted = len(text) - max_chars
    return f"{text[:head]}\n\n[... omitted {omitted} chars ...]\n\n{text[-tail:]}"


def stable_example_id(file_name: str, message_id: str, index: int) -> str:
    key = f"{file_name}:{message_id}:{index}".encode("utf-8", errors="replace")
    return hashlib.sha1(key).hexdigest()[:16]


def extract_text_parts(parts: Any, stats: ConversionStats, max_chars: int) -> str:
    if isinstance(parts, str):
        return clip_text(parts.strip(), max_chars)
    if not isinstance(parts, list):
        return ""

    out: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type == "text":
            value = str(part.get("text") or "").strip()
            if value:
                out.append(value)
        elif part_type == "image":
            stats.image_parts += 1
            out.append("[image omitted]")
        elif part_type == "thinking":
            stats.thinking_parts += 1
        elif part_type == "toolCall":
            continue
    return clip_text("\n".join(out).strip(), max_chars)


def convert_tool_call(part: dict[str, Any]) -> dict[str, Any] | None:
    name = part.get("name")
    if not name:
        return None
    arguments = part.get("arguments")
    if arguments is None:
        arguments = {}
    return {
        "id": str(
            part.get("id")
            or f"call_{hashlib.sha1(json.dumps(part, sort_keys=True, default=str).encode()).hexdigest()[:12]}"
        ),
        "type": "function",
        "function": {
            "name": str(name),
            "arguments": arguments,
        },
    }


def extract_assistant_message(
    raw_message: dict[str, Any], stats: ConversionStats, args: argparse.Namespace
) -> dict[str, Any] | None:
    parts = raw_message.get("content") or []
    text = extract_text_parts(parts, stats, args.max_assistant_chars)
    tool_calls: list[dict[str, Any]] = []
    reasoning: list[str] = []

    if isinstance(parts, list):
        for part in parts:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "toolCall":
                call = convert_tool_call(part)
                if call is not None:
                    tool_calls.append(call)
            elif args.include_reasoning and part.get("type") == "thinking":
                thinking = str(part.get("thinking") or "").strip()
                if thinking:
                    reasoning.append(thinking)

    if not text and not tool_calls:
        return None

    message: dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if reasoning:
        message["reasoning_content"] = "\n\n".join(reasoning)
    return message


def raw_event_to_chat_message(
    event: dict[str, Any], stats: ConversionStats, args: argparse.Namespace
) -> dict[str, Any] | None:
    if event.get("type") != "message":
        return None
    stats.message_events += 1
    raw_message = event.get("message") or {}
    role = raw_message.get("role")

    if role == "user":
        content = extract_text_parts(raw_message.get("content") or [], stats, args.max_user_chars)
        if not content:
            stats.skipped_messages += 1
            return None
        stats.user_messages += 1
        return {"role": "user", "content": content}

    if role == "assistant":
        message = extract_assistant_message(raw_message, stats, args)
        if message is None:
            stats.skipped_messages += 1
            return None
        stats.assistant_messages += 1
        return message

    if role == "toolResult":
        content = extract_text_parts(raw_message.get("content") or [], stats, args.max_tool_result_chars)
        if not content:
            content = "[empty tool result]"
        stats.tool_messages += 1
        return {
            "role": "tool",
            "tool_call_id": str(raw_message.get("toolCallId") or ""),
            "name": str(raw_message.get("toolName") or "unknown"),
            "content": content,
        }

    stats.skipped_messages += 1
    return None


def generic_tool_schema(name: str) -> dict[str, Any]:
    return {
        "description": f"Pi coding-agent tool named {name}.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }


def make_tool_definitions(tool_names: set[str]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for name in sorted(n for n in tool_names if n):
        schema = KNOWN_TOOL_SCHEMAS.get(name, generic_tool_schema(name))
        tools.append({"type": "function", "function": {"name": name, **schema}})
    return tools


def trim_context(messages: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    original = [copy.deepcopy(message) for message in messages]
    context = list(original)
    if args.max_context_messages > 0 and len(context) > args.max_context_messages:
        tail = context[-args.max_context_messages :]
        if not any(message.get("role") == "user" for message in tail):
            last_user_index = max(
                (index for index, message in enumerate(context) if message.get("role") == "user"),
                default=-1,
            )
            if last_user_index >= 0:
                user_anchor = context[last_user_index]
                tail = [user_anchor] + context[-max(1, args.max_context_messages - 1) :]
        context = tail

    if not any(message.get("role") == "user" for message in context):
        context = [
            {"role": "user", "content": "Continue the coding-agent session from the preceding context."}
        ] + context

    def compact_messages(rows: list[dict[str, Any]], max_chars: int) -> list[dict[str, Any]]:
        compacted = [copy.deepcopy(row) for row in rows]
        for row in compacted:
            if isinstance(row.get("content"), str):
                row["content"] = clip_text(row["content"], max_chars)
        return compacted

    for per_field_limit in (4000, 2000, 1000, 500):
        compacted = compact_messages(context, per_field_limit)
        if len(json.dumps(compacted, ensure_ascii=False, default=str)) <= args.max_prompt_chars:
            return coherent_context(compacted)
        context = compacted

    while (
        len(json.dumps(context, ensure_ascii=False, default=str)) > args.max_prompt_chars and len(context) > 2
    ):
        del context[1]
    return coherent_context(context)


def coherent_context(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: set[str] = set()
    result = []
    for message in messages:
        if message["role"] == "tool" and message.get("tool_call_id") not in calls:
            continue
        result.append(message)
        calls.update(call["id"] for call in message.get("tool_calls", []))
    return result


def load_raw_events(path: Path, stats: ConversionStats) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            stats.events += 1
            try:
                event = json.loads(line)
                if not isinstance(event, dict) or (
                    event.get("type") == "message" and not isinstance(event.get("message"), dict)
                ):
                    stats.invalid_events += 1
                    continue
                if event.get("type") == "message" and not isinstance(
                    event["message"].get("content", []), (list, str)
                ):
                    stats.invalid_events += 1
                    continue
                events.append(event)
            except json.JSONDecodeError:
                stats.json_errors += 1
    return events


def render_prompt_completion(
    processor: Any,
    context: list[dict[str, Any]],
    assistant_message: dict[str, Any],
    tools: list[dict[str, Any]],
) -> tuple[str, str]:
    kwargs = {
        "tokenize": False,
        "enable_thinking": False,
    }
    if tools:
        kwargs["tools"] = tools
    prompt = processor.apply_chat_template(context, add_generation_prompt=True, **kwargs)
    full = processor.apply_chat_template(context + [assistant_message], add_generation_prompt=False, **kwargs)
    if not full.startswith(prompt):
        raise ValueError("rendered full conversation does not start with rendered prompt")
    return prompt, full[len(prompt) :]


def build_examples(
    raw_dir: Path, processor: Any, args: argparse.Namespace
) -> tuple[list[dict[str, Any]], ConversionStats]:
    from jinja2 import TemplateError

    stats = ConversionStats()
    examples: list[dict[str, Any]] = []
    files = sorted(raw_dir.glob("*.jsonl"))
    if args.max_files > 0:
        files = files[: args.max_files]

    for file_index, path in enumerate(files):
        stats.files += 1
        events = load_raw_events(path, stats)
        session_ids = [str(e["id"]) for e in events if e.get("type") == "session" and e.get("id")]
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        group_id = session_ids[0] if session_ids else source_hash
        tool_names: set[str] = set()
        for event in events:
            if event.get("type") != "message":
                continue
            raw_message = event.get("message") or {}
            for part in raw_message.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "toolCall" and part.get("name"):
                    tool_names.add(str(part["name"]))
        tools = make_tool_definitions(tool_names)

        conversation: list[dict[str, Any]] = []
        for message_index, event in enumerate(events):
            message = raw_event_to_chat_message(event, stats, args)
            if message is None:
                continue

            if message["role"] == "assistant" and any(m.get("role") == "user" for m in conversation):
                context = trim_context(conversation, args)
                try:
                    prompt, completion = render_prompt_completion(processor, context, message, tools)
                except (ValueError, TypeError, KeyError, IndexError, TemplateError):
                    stats.skipped_render_errors += 1
                else:
                    if not completion.strip():
                        stats.skipped_empty_completion += 1
                    elif not prompt:
                        stats.skipped_prefix_mismatch += 1
                    else:
                        prompt_ids = processor(prompt, add_special_tokens=False)["input_ids"]
                        full_ids = processor(prompt + completion, add_special_tokens=False)["input_ids"]
                        if full_ids[: len(prompt_ids)] != prompt_ids or len(full_ids) <= len(prompt_ids):
                            stats.skipped_token_prefix += 1
                            conversation.append(message)
                            continue
                        examples.append(
                            {
                                "id": stable_example_id(path.name, str(event.get("id") or ""), message_index),
                                "source_file": path.name,
                                "group_id": group_id,
                                "source_sha256": source_hash,
                                "prompt": prompt,
                                "completion": completion,
                            }
                        )
                        stats.assistant_examples += 1
                        if args.max_examples > 0 and len(examples) >= args.max_examples:
                            return examples, stats

            conversation.append(message)

        if (file_index + 1) % 100 == 0:
            print(f"phase=convert files={file_index + 1} examples={len(examples)}", flush=True)

    return examples, stats


def local_raw_data(args: argparse.Namespace) -> Path:
    path = local_directory(args.raw_dir, "Raw trace")
    if not list(path.glob("*.jsonl")):
        raise FileNotFoundError(f"No raw session *.jsonl files in {path}")
    return path


def load_processor(model_id: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, local_files_only=True, trust_remote_code=False)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def tokenization_check(
    processor: Any, examples: list[dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    tokenizer = getattr(processor, "tokenizer", processor)
    sample = examples[: min(32, len(examples))]
    lengths: list[int] = []
    completion_lengths: list[int] = []
    for row in sample:
        encoded = tokenizer(row["prompt"] + row["completion"], add_special_tokens=False)
        comp_encoded = tokenizer(row["completion"], add_special_tokens=False)
        lengths.append(len(encoded["input_ids"]))
        completion_lengths.append(len(comp_encoded["input_ids"]))
    over = sum(length > args.max_length for length in lengths)
    return {
        "checked": len(sample),
        "max_length": args.max_length,
        "min_tokens": min(lengths) if lengths else 0,
        "max_tokens": max(lengths) if lengths else 0,
        "over_max_length": over,
        "min_completion_tokens": min(completion_lengths) if completion_lengths else 0,
        "max_completion_tokens": max(completion_lengths) if completion_lengths else 0,
    }


def filter_examples_by_length(
    processor: Any, examples: list[dict[str, Any]], args: argparse.Namespace
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not args.filter_overlength:
        return examples, {"enabled": False, "kept": len(examples), "dropped": 0}

    tokenizer = getattr(processor, "tokenizer", processor)
    kept: list[dict[str, Any]] = []
    dropped = 0
    lengths: list[int] = []
    for index, row in enumerate(examples):
        encoded = tokenizer(row["prompt"] + row["completion"], add_special_tokens=False)
        length = len(encoded["input_ids"])
        lengths.append(length)
        if length <= args.max_length:
            kept.append(row)
        else:
            dropped += 1
        if (index + 1) % 2000 == 0:
            print(
                f"phase=filter_lengths checked={index + 1} kept={len(kept)} dropped={dropped}",
                flush=True,
            )

    return kept, {
        "enabled": True,
        "max_length": args.max_length,
        "checked": len(examples),
        "kept": len(kept),
        "dropped": dropped,
        "min_tokens": min(lengths) if lengths else 0,
        "max_tokens": max(lengths) if lengths else 0,
    }


def split_rows(examples: list[dict[str, Any]], args: argparse.Namespace):
    groups_override = json.loads(Path(args.split_groups).read_text()) if args.split_groups else {}
    if not isinstance(groups_override, dict):
        raise TypeError("--split-groups must contain a filename-to-group JSON object")
    # Union provenance groups with byte-identical source files before assigning splits.
    parent = {}

    def find(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]

    for row in examples:
        group = str(groups_override.get(row["source_file"], row["group_id"]))
        a, b = find("group:" + group), find("hash:" + row["source_sha256"])
        parent[max(a, b)] = min(a, b)
    grouped = {}
    for row in examples:
        key = find("hash:" + row["source_sha256"])
        grouped.setdefault(key, []).append(row)
    keys = sorted(grouped, key=lambda k: hashlib.sha256(f"{args.seed}:{k}".encode()).hexdigest())
    if args.eval_size and len(keys) < 2:
        raise ValueError(
            "Validation requires at least two independent session groups; supply more traces or --eval-size 0"
        )
    target = min(args.eval_size, max(1, len(examples) // 20)) if args.eval_size else 0
    eval_keys, size = set(), 0
    for key in keys[:-1]:
        if size >= target:
            break
        eval_keys.add(key)
        size += len(grouped[key])
    train = [r for k in keys if k not in eval_keys for r in grouped[k]]
    evaluation = [r for k in keys if k in eval_keys for r in grouped[k]]
    manifest = {
        "seed": args.seed,
        "policy": "session-and-duplicate-groups-v1",
        "validation_target_rows": target,
        "train": [
            {
                "id": r["id"],
                "source_file": r["source_file"],
                "group": find("hash:" + r["source_sha256"]),
                "source_sha256": r["source_sha256"],
            }
            for r in train
        ],
        "validation": [
            {
                "id": r["id"],
                "source_file": r["source_file"],
                "group": find("hash:" + r["source_sha256"]),
                "source_sha256": r["source_sha256"],
            }
            for r in evaluation
        ],
    }
    return train, evaluation, manifest


def make_dataset_splits(examples: list[dict[str, Any]], args: argparse.Namespace):
    from datasets import Dataset

    train, evaluation, manifest = split_rows(examples, args)
    path = Path(args.output_dir) / "split_manifest.json"
    if args.resume_from_checkpoint and (not path.is_file() or json.loads(path.read_text()) != manifest):
        raise ValueError("Resume requires the original unchanged split manifest, data and seed")
    write_json(path, manifest)
    return Dataset.from_list(train), Dataset.from_list(evaluation)


def resolve_lora_target_modules(model: Any, args: argparse.Namespace) -> list[str]:
    if args.target_modules_regex and not args.target_modules:
        pattern = re.compile(args.target_modules_regex)
        matched_modules = [
            (name, module) for name, module in model.named_modules() if pattern.fullmatch(name)
        ]
        matches = [name for name, _module in matched_modules]
        if not matches:
            raise RuntimeError(f"no LoRA target modules matched regex: {args.target_modules_regex}")
        sample = [{"name": name, "type": type(module).__name__} for name, module in matched_modules[:8]]
        print(
            f"phase=lora_targets mode=regex count={len(matches)} sample={json.dumps(sample, sort_keys=True)}",
            flush=True,
        )
        return matches

    modules = [part.strip() for part in args.target_modules.split(",") if part.strip()]
    if not modules:
        raise ValueError("no LoRA target modules configured")
    print(
        f"phase=lora_targets mode=suffix count={len(modules)} modules={json.dumps(modules, sort_keys=True)}",
        flush=True,
    )
    return modules


def train(
    examples: list[dict[str, Any]], processor: Any, stats: ConversionStats, args: argparse.Namespace
) -> None:
    import torch
    import trackio
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    def finish_trackio_safely() -> None:
        try:
            trackio.finish()
        except RuntimeError as exc:
            if "Call trackio.init() before trackio.finish()" in str(exc):
                return
            print(f"phase=trackio_finish_warning type={type(exc).__name__} message={exc}", flush=True)
        except Exception as exc:  # noqa: BLE001 - best-effort cleanup also runs during exception unwinding
            print(f"phase=trackio_finish_warning type={type(exc).__name__} message={exc}", flush=True)

    os.environ["TRACKIO_PROJECT"] = args.trackio_project
    os.environ.setdefault("TRACKIO_DIR", str(Path(args.output_dir) / "trackio"))

    trackio_config = {
        "model": args.model_path,
        "dataset": str(Path(args.raw_dir).resolve()),
        "method": "sft_lora",
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "max_steps": args.max_steps,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules_regex": args.target_modules_regex,
        "target_modules": args.target_modules,
        "completion_only_loss": True,
    }
    print(
        "phase=trackio_init "
        f"project={args.trackio_project} run={args.run_name} "
        f"group={args.trackio_group} storage=local",
        flush=True,
    )
    trackio.init(
        project=args.trackio_project,
        name=args.run_name,
        group=args.trackio_group,
        space_id=None,
        resume="allow" if args.resume_from_checkpoint else "never",
        config=trackio_config,
    )
    atexit.register(finish_trackio_safely)

    class LocalTrackingCallback(TrainerCallback):
        def on_log(self, training_args, state, control, logs=None, **kwargs):
            if logs:
                trackio.log(logs, step=state.global_step)

    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    print("phase=load_model", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if args.bf16 else "auto",
        device_map={"": args.device},
        local_files_only=True,
        trust_remote_code=False,
        quantization_config=quantization_config,
    )
    model.config.use_cache = False

    train_ds, eval_ds = make_dataset_splits(examples, args)
    lora_target_modules = resolve_lora_target_modules(model, args)
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_target_modules,
    )

    optim = "paged_adamw_8bit" if args.load_in_4bit else "adamw_torch"
    training_args = SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_length,
        completion_only_loss=True,
        packing=False,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if len(eval_ds) else "no",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        optim=optim,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,
        report_to=[],
        run_name=args.run_name,
        push_to_hub=False,
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=True,
    )

    # The pinned Trainer otherwise counts all visible GPUs and changes batch size.
    # This template deliberately supports one selected device, not DataParallel.
    training_args._n_gpu = 1

    write_json(
        Path(args.output_dir) / "conversion_stats.json",
        {
            **stats.__dict__,
            "model_id": args.model_path,
            "dataset_id": str(Path(args.raw_dir).resolve()),
            "train_examples": len(train_ds),
            "eval_examples": len(eval_ds),
            "max_length": args.max_length,
            "completion_only_loss": True,
            "include_reasoning": args.include_reasoning,
        },
    )

    print(
        "phase=train_start "
        f"train_examples={len(train_ds)} eval_examples={len(eval_ds)} "
        f"model={args.model_path} device={args.device}",
        flush=True,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if len(eval_ds) else None,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=[LocalTrackingCallback()],
    )
    # Inspect the real collator output before starting an expensive run.
    batch = next(iter(trainer.get_train_dataloader()))
    if batch["input_ids"].shape[0] > args.per_device_train_batch_size:
        raise ValueError("Trainer expanded the batch beyond the explicitly selected single-device profile")
    if not (batch["labels"] != -100).any() or not (batch["labels"] == -100).any():
        raise ValueError("Expected completion targets and masked prompt/padding tokens")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(args.output_dir)
    trainer.save_state()
    tokenizer.save_pretrained(args.output_dir)
    metrics = dict(train_result.metrics)
    metrics["train_examples"] = len(train_ds)
    metrics["eval_examples"] = len(eval_ds)
    if args.device.startswith("cuda:"):
        metrics["peak_gpu_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
        metrics["peak_gpu_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(args.device)
    trainer.save_metrics("train", metrics)

    if len(eval_ds):
        trainer.save_metrics("eval", trainer.evaluate())
    write_json(
        Path(args.output_dir) / "artifact_manifest.json",
        {
            "base_model_path": str(Path(args.model_path).resolve()),
            "base_model_files": json.loads((Path(args.output_dir) / "input_assets.json").read_text())[
                "model_files"
            ],
            "adapter_files": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(args.output_dir).glob("adapter*")
                if p.is_file()
            },
        },
    )

    finish_trackio_safely()
    atexit.unregister(finish_trackio_safely)
    print("phase=done", json.dumps(metrics, sort_keys=True), flush=True)


def main() -> None:
    args = parse_args()
    configure_local_runtime(args.runtime_dir)
    validate_model(args.model_path)
    random.seed(args.seed)
    raw_dir = local_raw_data(args)
    if not args.prepare_only:
        validate_device(args.device, args.load_in_4bit)
        output = Path(args.output_dir)
        if output.exists() and any(output.iterdir()) and not args.resume_from_checkpoint:
            raise FileExistsError(f"Use a new output directory or --resume-from-checkpoint: {output}")
        if args.resume_from_checkpoint:
            checkpoint = local_directory(args.resume_from_checkpoint, "Checkpoint")
            if (
                not checkpoint.is_relative_to(output.resolve())
                or not (checkpoint / "trainer_state.json").is_file()
            ):
                raise ValueError(
                    "Resume checkpoint must belong to this output directory and include trainer state"
                )
            for filename in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
                if not (checkpoint / filename).is_file():
                    raise FileNotFoundError(f"Incomplete resume checkpoint: {filename}")
            previous_step = json.loads((checkpoint / "trainer_state.json").read_text())["global_step"]
            if args.max_steps > 0 and args.max_steps <= previous_step:
                raise ValueError("Resume --max-steps must exceed the saved global step")
        current = {k: v for k, v in vars(args).items() if k not in ("resume_from_checkpoint", "max_steps")}
        config_path = output / "run_config.json"
        if args.resume_from_checkpoint and (
            not config_path.exists() or json.loads(config_path.read_text()) != current
        ):
            raise ValueError("Resume configuration differs from original run")
        assets = {
            "model_files": file_manifest(Path(args.model_path)),
            "raw_files": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(raw_dir.glob("*.jsonl"))
            },
        }
        assets_path = output / "input_assets.json"
        if args.resume_from_checkpoint and (
            not assets_path.exists() or json.loads(assets_path.read_text()) != assets
        ):
            raise ValueError("Resume requires unchanged model and source data hashes")
        write_json(assets_path, assets)
        write_json(config_path, current)
        write_json(
            output / ("resume_invocation.json" if args.resume_from_checkpoint else "invocation.json"),
            {
                "argv": sys.argv,
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in (
                        "torch",
                        "transformers",
                        "trl",
                        "peft",
                        "datasets",
                        "trackio",
                        "accelerate",
                        "bitsandbytes",
                    )
                },
            },
        )
    processor = load_processor(args.model_path)
    examples, stats = build_examples(raw_dir, processor, args)
    examples, length_filter = filter_examples_by_length(processor, examples, args)
    if len(examples) < 2:
        raise RuntimeError(f"not enough examples after conversion: {len(examples)}")

    work_dir = Path(args.work_dir)
    write_jsonl(work_dir / "prepared_examples.sample.jsonl", examples[: min(100, len(examples))])
    summary: dict[str, Any] = {
        **stats.__dict__,
        "model_id": args.model_path,
        "dataset_id": str(Path(args.raw_dir).resolve()),
        "raw_dir": str(raw_dir),
        "examples": len(examples),
        "length_filter": length_filter,
        "include_reasoning": args.include_reasoning,
    }
    if not args.skip_tokenizer_check:
        summary["tokenization_check"] = tokenization_check(processor, examples, args)
    write_json(work_dir / "conversion_summary.json", summary)
    print("phase=conversion_summary", json.dumps(summary, sort_keys=True), flush=True)

    if args.prepare_only:
        return

    train(examples, processor, stats, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"phase=error type={type(exc).__name__} message={exc}", file=sys.stderr, flush=True)
        raise
