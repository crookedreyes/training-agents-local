"""Shared local-only runtime policy for the lightweight training example."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import socket
import sys
from pathlib import Path

_POLICY_INSTALLED = False


def configure_local_runtime(root: str | Path = "workspaces/local-agent") -> Path:
    """Apply before importing ML/UI libraries. No network asset fallback is allowed."""
    global _POLICY_INSTALLED
    forbidden = (
        "TRACKIO_SPACE_ID",
        "TRACKIO_SERVER_URL",
        "TRACKIO_WEBHOOK_URL",
        "TRACKIO_BUCKET_ID",
        "TRACKIO_DATASET_ID",
        "TRACKIO_LOGO_LIGHT_URL",
        "TRACKIO_LOGO_DARK_URL",
    )
    present = [name for name in forbidden if os.environ.get(name)]
    if present:
        raise ValueError("Remove remote tracking settings before local execution: " + ", ".join(present))
    if "/" in os.environ.get("TRACKIO_THEME", ""):
        raise ValueError("TRACKIO_THEME must be a built-in local theme")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name, value in {
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        "GRADIO_ANALYTICS_ENABLED": "False",
        "WANDB_DISABLED": "true",
        "TRACKIO_DIR": str(root / "trackio"),
        "HF_HOME": str(root / "cache/huggingface"),
        "TMPDIR": str(root / "tmp"),
        "TRITON_CACHE_DIR": str(root / "cache/triton"),
        "TORCH_HOME": str(root / "cache/torch"),
        "XDG_CACHE_HOME": str(root / "cache"),
    }.items():
        os.environ[name] = value
    (root / "tmp").mkdir(exist_ok=True)
    if not _POLICY_INSTALLED:
        sys.addaudithook(_network_audit)
        _POLICY_INSTALLED = True
    return root


def _loopback(host: str | bytes | None) -> bool:
    if isinstance(host, bytes):
        host = host.decode("ascii", errors="replace")
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _network_audit(event: str, args: tuple) -> None:
    # This guards Python network calls, including attempted DNS lookups. Native
    # libraries/subprocesses additionally need OS network isolation for proof.
    if event == "socket.getaddrinfo" and not _loopback(args[0]):
        raise PermissionError("Local execution prohibits external DNS lookup")
    if event in ("socket.connect", "socket.sendto"):
        sock, address = args[0], args[-1]
        if sock.family in (socket.AF_INET, socket.AF_INET6) and not _loopback(address[0]):
            raise PermissionError("Local execution prohibits external network connections")


def local_directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"{label} directory is missing: {path}. Import it during setup; runtime will not download it."
        )
    return path


def validate_model(value: str | Path) -> Path:
    path = local_directory(value, "Model")
    for name in ("config.json", "tokenizer_config.json"):
        if not (path / name).is_file():
            raise FileNotFoundError(f"Model asset missing: {path / name}")
    if not any((path / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "spiece.model")):
        raise FileNotFoundError(f"No tokenizer vocabulary in {path}")
    indices = list(path.glob("*.safetensors.index.json")) + list(path.glob("pytorch_model*.index.json"))
    if indices:
        for index in indices:
            mapping = json.loads(index.read_text()).get("weight_map", {})
            if not mapping:
                raise ValueError(f"Empty weight index: {index}")
            for name in set(mapping.values()):
                shard = (path / name).resolve()
                if not shard.is_relative_to(path) or not shard.is_file():
                    raise FileNotFoundError(f"Model shard missing or outside model directory: {name}")
    elif not (list(path.glob("model*.safetensors")) or list(path.glob("pytorch_model*.bin"))):
        raise FileNotFoundError(f"No Transformers weights in {path}; GGUF alone is not a training checkpoint")
    return path


def validate_device(device: str, quantized: bool = True) -> None:
    import torch

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError(
            "This template currently validates one GPU per process; distributed training is not enabled"
        )
    if device == "cpu":
        if quantized:
            raise ValueError("CPU smoke requires --no-load-in-4bit and --no-bf16")
        return
    if not device.startswith("cuda:") or not device[5:].isdigit():
        raise ValueError("Use an explicit device: cuda:0, cuda:1, or cpu")
    if not torch.cuda.is_available() or int(device[5:]) >= torch.cuda.device_count():
        raise RuntimeError(
            f"GPU unavailable: {device}. Check driver and device access outside the agent sandbox."
        )
    torch.cuda.set_device(device)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_manifest(path: Path) -> dict[str, str]:
    return {
        str(p.relative_to(path)): sha256(p)
        for p in sorted(path.rglob("*"))
        if p.is_file() and ".cache" not in p.parts
    }


def adapter_manifest(path: Path) -> dict[str, str]:
    """Hash adapter payload, not sibling checkpoints or nested evaluation logs."""
    names = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    )
    return {name: sha256(path / name) for name in names if (path / name).is_file()}


def load_local_model(model_path: str, adapter_path: str | None, device: str, quantized: bool, bf16: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    path = validate_model(model_path)
    validate_device(device, quantized)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    dtype = torch.bfloat16 if bf16 and device != "cpu" else torch.float32
    quantization = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        if quantized
        else None
    )
    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=dtype,
        device_map={"": device},
        quantization_config=quantization,
    )
    if adapter_path:
        from peft import PeftModel

        adapter = local_directory(adapter_path, "Adapter")
        model = PeftModel.from_pretrained(model, adapter, local_files_only=True, is_trainable=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 256, temperature: float = 0.0) -> str:
    import torch

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    kwargs = {
        "do_sample": temperature > 0,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        kwargs["temperature"] = temperature
    model.eval()
    model.config.use_cache = True
    with torch.inference_mode():
        result = model.generate(**inputs, **kwargs)
    return tokenizer.decode(result[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
