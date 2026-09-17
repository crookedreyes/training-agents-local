# Training On Agent Traces

Hugging Face Hub supports viewing raw JSONL traces from Claude Code, Codex, and
Pi Agent. Local session locations include:

- Codex: `~/.codex/sessions`
- Pi: `~/.pi/agent/sessions`
- Claude Code: `~/.claude/projects`

The Hub docs warn that traces can contain prompts, tool inputs, command output,
local paths, screenshots, secrets, private code, and personal data. Review and
redact traces before public upload or training.

Copy only the selected reviewed trace files into a local dataset directory.
Do not upload traces or read every local agent session automatically. Convert
bespoke formats into the local template schema before training. The existing
converter accepts Pi-Mono raw sessions; other formats need a separate converter.

Use trace data only after deciding what the student should imitate:

- final assistant answers
- tool-call decisions
- repair behavior after failed commands
- concise summaries of tool results
- full transcripts, if and only if every role/token is appropriate as target
  behavior

If trace rows are not already in `messages` or `prompt`/`completion` format,
preprocess them before calling `SFTTrainer`.
