"""Run a logged synthetic GPU SFT/save/resume/reload smoke. No downloads."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    example = Path(__file__).resolve().parents[1]
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    run = out / "adapter"
    base = "workspaces/local-agent/fixtures/model"
    raw = "workspaces/local-agent/fixtures/traces"
    shared = ["--model-path", base, "--raw-dir", raw, "--device", args.device]
    training = [
        sys.executable,
        str(example / "train_sft.py"),
        *shared,
        "--output-dir",
        str(run),
        "--work-dir",
        str(out / "prepared"),
        "--max-length",
        "128",
        "--gradient-accumulation-steps",
        "1",
        "--logging-steps",
        "1",
        "--eval-steps",
        "1",
        "--save-steps",
        "1",
        "--target-modules",
        "q_proj,v_proj",
        "--eval-size",
        "1",
        "--run-name",
        out.name,
    ]
    commands = [
        [sys.executable, str(example / "preflight.py"), *shared],
        [*training, "--max-steps", "2"],
        [*training, "--max-steps", "4", "--resume-from-checkpoint", str(run / "checkpoint-2")],
        [
            sys.executable,
            str(example / "generate.py"),
            "--model-path",
            base,
            "--adapter-path",
            str(run),
            "--device",
            args.device,
            "--prompt",
            "hello",
            "--max-new-tokens",
            "8",
        ],
    ]
    for i, command in enumerate(commands):
        (out / f"{i}-command.json").write_text(json.dumps(command, indent=2))
        with (out / f"{i}-output.log").open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        print(
            f"phase=smoke step={i} returncode={result.returncode} log={out / f'{i}-output.log'}", flush=True
        )
        if result.returncode:
            raise SystemExit(result.returncode)
    state = json.loads((run / "trainer_state.json").read_text())
    assert state["global_step"] == 4
    assert any(row.get("grad_norm", 0) > 0 for row in state["log_history"])
    manifest = json.loads((run / "split_manifest.json").read_text())
    assert {r["group"] for r in manifest["train"]}.isdisjoint({r["group"] for r in manifest["validation"]})
    print("Synthetic SFT/resume/reload smoke passed. This is not a model-quality result.")


if __name__ == "__main__":
    main()
