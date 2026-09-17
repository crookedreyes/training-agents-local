# Local job execution

Use explicit workstation GPU assignments and local process IDs. Start with one
GPU, a small token budget and a 1–5 step smoke. Keep caches, outputs and temporary
data on disk under ignored workspaces. Do not launch HF Jobs or require a cluster.

Record command, versions, input hashes, split, local Trackio directory, checkpoint
and evaluation paths. Save before post-eval and expose a tested resume command.
Use `rg` and local process status to diagnose failures before rerunning.
