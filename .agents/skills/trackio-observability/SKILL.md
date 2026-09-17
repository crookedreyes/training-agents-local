---
name: trackio-observability
description: Use when instrumenting or inspecting TRL training runs with Trackio, run names, metric schemas, dashboards, logs, grep or ripgrep, local process status, local artifacts, or experiment result summaries.
---

# Trackio Observability

Use this skill to make training runs observable and debuggable.

## Workflow

1. Define the run identity: project, run name, method, model, dataset, seed, and
   challenge.
2. Use local Trackio for training runs and a loopback-only dashboard. Reject
   inherited Space, remote server, bucket, dataset and webhook destinations.
   Configure local storage before importing Trackio. See the repository guide `docs/local-execution.md` (path relative to the repo root).
3. Make logs grep-friendly with clear phase markers.
4. Persist artifacts intentionally: model, adapter, config, metrics, traces, and
   evaluation outputs.
5. Inspect local state with Trackio, process status and `rg` over local artifacts.

## Reporting Shape

Return:

- run id or job id
- local Trackio dashboard and data directory
- command or script inspected
- latest metrics
- artifact paths
- failure signatures
- next minimal action

Never print tokens, secrets, private credentials, or full logs unless the user
explicitly asks for a raw excerpt.

## References

- `references/tracking-schema.md`: run metadata and metric schema.
- `references/log-inspection.md`: local logs and artifact triage.
