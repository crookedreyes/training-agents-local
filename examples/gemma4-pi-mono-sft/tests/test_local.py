import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE))
from local_runtime import validate_model
from train_sft import ConversionStats, coherent_context, load_raw_events, split_rows


def rows():
    return [
        {
            "id": f"{g}-{i}",
            "source_file": f"{g}.jsonl",
            "group_id": g,
            "source_sha256": f"hash-{g}",
            "prompt": "user",
            "completion": "assistant",
        }
        for g in ["a", "b", "c"]
        for i in range(20)
    ]


def args(**kwargs):
    return argparse.Namespace(seed=42, eval_size=3, split_groups=None, **kwargs)


def test_session_split_disjoint_and_order_independent():
    train, validation, manifest = split_rows(rows(), args())
    assert {r["source_file"] for r in train}.isdisjoint({r["source_file"] for r in validation})
    assert len(train) + len(validation) == 60
    train2, val2, _ = split_rows(list(reversed(rows())), args())
    assert {r["id"] for r in train} == {r["id"] for r in train2}
    assert {r["id"] for r in validation} == {r["id"] for r in val2}
    assert manifest["policy"] == "session-and-duplicate-groups-v1"


def test_duplicate_files_and_related_groups_stay_together(tmp_path):
    examples = rows()
    examples.append({**examples[0], "id": "copy", "source_file": "copy.jsonl", "group_id": "different"})
    group_path = tmp_path / "groups.json"
    group_path.write_text(json.dumps({"a.jsonl": "same-task", "b.jsonl": "same-task"}))
    options = args()
    options.split_groups = str(group_path)
    train, validation, _ = split_rows(examples, options)
    for split in (train, validation):
        files = {r["source_file"] for r in split}
        assert bool(files & {"a.jsonl", "b.jsonl", "copy.jsonl"}) == {
            "a.jsonl",
            "b.jsonl",
            "copy.jsonl",
        }.issubset(files)


def test_single_group_fails_unless_no_eval():
    data = rows()[:20]
    with pytest.raises(ValueError, match="two independent"):
        split_rows(data, args())
    options = args()
    options.eval_size = 0
    train, validation, _ = split_rows(data, options)
    assert len(train) == 20 and not validation


def test_bad_json_shapes_counted(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('null\n[]\n{"type":"message","message":[]}\ninvalid\n{"type":"session","id":"ok"}\n')
    stats = ConversionStats()
    assert len(load_raw_events(path, stats)) == 1
    assert stats.invalid_events == 3 and stats.json_errors == 1


def test_tool_arguments_preserved_and_orphans_removed():
    call = {"id": "a", "function": {"name": "write", "arguments": {"content": "x" * 10_000}}}
    context = [
        {"role": "tool", "tool_call_id": "missing", "content": "bad"},
        {"role": "assistant", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "a", "content": "ok"},
    ]
    result = coherent_context(context)
    assert len(result) == 2 and result[0]["tool_calls"][0] == call


def test_missing_model_shard_fails_without_download(tmp_path):
    for name in ["config.json", "tokenizer_config.json", "tokenizer.json"]:
        (tmp_path / name).write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "missing.safetensors"}})
    )
    with pytest.raises(FileNotFoundError, match="shard"):
        validate_model(tmp_path)


def test_network_policy_and_remote_environment(tmp_path):
    script = f"""import sys, socket, os
sys.path.insert(0, {str(EXAMPLE)!r})
from local_runtime import configure_local_runtime
configure_local_runtime({str(tmp_path)!r})
assert os.environ['HF_HUB_OFFLINE'] == '1'
try:
    socket.getaddrinfo('example.com', 443)
except PermissionError:
    pass
else:
    raise AssertionError('external DNS allowed')
try:
    socket.create_connection(('1.1.1.1', 443))
except PermissionError:
    pass
else:
    raise AssertionError('external connection allowed')
socket.getaddrinfo('127.0.0.1', 80)
"""
    subprocess.run([sys.executable, "-c", script], check=True)
    import os

    env = {**os.environ, "TRACKIO_SPACE_ID": "remote/test"}
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode != 0 and "Remove remote tracking settings" in result.stderr


def test_negative_eval_rejected_before_imports():
    result = subprocess.run(
        [
            sys.executable,
            str(EXAMPLE / "train_sft.py"),
            "--model-path",
            "missing",
            "--raw-dir",
            "missing",
            "--eval-size",
            "-1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2 and "cannot be negative" in result.stderr


def test_adapter_manifest_excludes_prior_eval_logs(tmp_path):
    from local_runtime import adapter_manifest

    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"fixture")
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval/protocol.json").write_text("old evaluation")
    assert set(adapter_manifest(tmp_path)) == {"adapter_config.json", "adapter_model.safetensors"}
