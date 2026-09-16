"""Check the packaged dashboard HTML/assets under external network isolation."""

import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

root = Path("workspaces/local-agent/runs/dashboard-check")
root.mkdir(parents=True, exist_ok=True)
command = [sys.executable, "examples/gemma4-pi-mono-sft/dashboard.py", "--port", "17860"]
(root / "command.json").write_text(json.dumps(command))
with (root / "output.log").open("w") as stream:
    proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
    try:
        for attempt in range(40):
            if proc.poll() is not None:
                raise RuntimeError((root / "output.log").read_text())
            try:
                html = urllib.request.urlopen("http://127.0.0.1:17860", timeout=1).read().decode()
                break
            except OSError:
                time.sleep(0.25)
        else:
            raise RuntimeError("Dashboard did not start")
        assets = re.findall(r'(?:src|href)="([^"#]+\.(?:js|css)(?:\?[^" ]*)?)"', html)
        checked = []
        for asset in assets:
            if asset.startswith(("http://", "https://", "//")):
                raise RuntimeError(f"External frontend dependency: {asset}")
            url = "http://127.0.0.1:17860/" + asset.lstrip("/")
            with urllib.request.urlopen(url, timeout=5) as result:
                assert result.status == 200
                checked.append(asset)
        assert checked, "No packaged JS/CSS found"
        (root / "result.json").write_text(json.dumps({"status": "ok", "assets": checked}, indent=2))
        print("Dashboard and bundled assets load locally:", checked)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
