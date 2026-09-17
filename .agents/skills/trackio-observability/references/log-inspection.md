# Local log inspection

```bash
rg -n "ERROR|Traceback|CUDA|OOM|reward|eval|checkpoint" workspaces/local-agent/runs/
```

Inspect local Trackio data, process status and artifacts. Report the command,
exit status, latest metric/checkpoint and minimal next action. Do not emit secrets
or full traces. Remote log inspection is not part of the local runtime.
