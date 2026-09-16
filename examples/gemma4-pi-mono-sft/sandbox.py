"""Execute benchmark code in an offline, disposable local sandbox."""

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Result:
    success: bool
    returncode: int
    output: str
    timed_out: bool = False


def check_sandbox(backend: str = "bubblewrap", image: str = "local-agent-eval:1"):
    if backend == "bubblewrap":
        if not shutil.which("bwrap"):
            raise RuntimeError(
                "Install bubblewrap during setup, or use --sandbox docker with a prepared image"
            )
        result = execute('print("sandbox-ok")', backend=backend, timeout=5, image=image)
        if not result.success:
            raise RuntimeError("Bubblewrap is unavailable in this execution context: " + result.output[-500:])
    elif backend == "docker":
        if not shutil.which("docker"):
            raise RuntimeError(
                "Docker is missing; install it and import/build the local eval image during setup"
            )
        subprocess.run(
            ["docker", "image", "inspect", image], check=True, capture_output=True, timeout=15
        )
    else:
        raise ValueError("Supported local sandboxes: bubblewrap, docker")


def _process_budget():
    # RLIMIT_NPROC counts existing tasks for this UID, including Torch threads.
    # Reserve a bounded additional allowance rather than failing on a busy host.
    count = 0
    for path in Path("/proc").glob("[0-9]*"):
        try:
            if path.stat().st_uid == os.getuid():
                count += len(list((path / "task").iterdir()))
        except (FileNotFoundError, PermissionError):
            continue
    return count + 64


def execute(
    code: str, backend: str = "bubblewrap", timeout: int = 30, image: str = "local-agent-eval:1"
) -> Result:
    import tempfile
    import uuid

    container = "local-eval-" + uuid.uuid4().hex
    if backend == "bubblewrap":
        command = [
            "bwrap",
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--ro-bind",
            "/usr",
            "/usr",
            "--symlink",
            "usr/bin",
            "/bin",
            "--symlink",
            "usr/lib",
            "/lib",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--chdir",
            "/tmp",
            "--clearenv",
            "--setenv",
            "PATH",
            "/usr/bin:/bin",
        ]
        if Path("/usr/lib64").exists():
            command += ["--symlink", "usr/lib64", "/lib64"]
        command += ["/usr/bin/python3", "-I", "-"]
        command = [
            "prlimit",
            "--as=1073741824",
            "--core=0",
            "--fsize=8388608",
            "--nofile=64",
            "--cpu=30",
            f"--nproc={_process_budget()}",
            "--",
            *command,
        ]
    elif backend == "docker":
        command = [
            "docker",
            "run",
            "--rm",
            "--pull=never",
            "--name",
            container,
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--memory=1g",
            "--cpus=1",
            "--pids-limit=64",
            "--user=65534:65534",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "-i",
            image,
            "python3",
            "-I",
            "-",
        ]
    else:
        raise ValueError("Unsupported sandbox backend")
    # Bound output on disk rather than allowing an untrusted program to fill RAM.
    with tempfile.TemporaryFile() as output:
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=output,
                stderr=output,
                start_new_session=True,
            )
            try:
                process.communicate(code.encode(), timeout=timeout)
                timed_out = False
            except subprocess.TimeoutExpired:
                import os
                import signal

                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                timed_out = True
            output.seek(0)
            text = output.read(16_384).decode(errors="replace")
            return Result(process.returncode == 0 and not timed_out, process.returncode, text, timed_out)
        finally:
            if backend == "docker":
                subprocess.run(
                    ["docker", "rm", "-f", container], capture_output=True, timeout=15, check=False
                )
