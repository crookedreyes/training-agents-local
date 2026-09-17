"""Opt-in integration checks requiring a host that permits Bubblewrap namespaces."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytestmark = pytest.mark.skipif(
    os.environ.get("LOCAL_AGENT_SANDBOX_TEST") != "1", reason="Requires explicit host namespace access"
)


def test_isolated_verifier_and_timeout():
    from sandbox import check_sandbox, execute

    check_sandbox()
    assert execute("assert 1 + 1 == 2").success
    assert not execute("assert False").success
    assert execute("while True: pass", timeout=1).timed_out
    assert execute("import os; assert not os.path.exists('/home/norte-ai')").success
    assert not execute("import socket; socket.create_connection(('1.1.1.1', 443), timeout=1)").success


@pytest.mark.parametrize("benchmark", ["humaneval", "mbpp"])
def test_original_inspect_scorer_accepts_correct_fixture(tmp_path, benchmark):
    from local_runtime import configure_local_runtime

    configure_local_runtime(tmp_path / "runtime")
    from inspect_ai import eval
    from inspect_ai.model import ModelOutput
    from inspect_ai.solver import solver

    from evaluate import local_task

    @solver
    def correct_completion():
        async def solve(state, generate):
            state.output = ModelOutput.from_content("fixture", "```python\ndef answer():\n    return 42\n```")
            return state

        return solve

    with local_task(benchmark, Path("workspaces/local-agent/fixtures/eval"), "bubblewrap", "") as task:
        task.solver = correct_completion()
        logs = eval(
            task,
            model="mockllm/model",
            log_dir=str(tmp_path / "logs"),
            max_samples=1,
            max_connections=1,
            display="none",
        )
    assert logs[0].status == "success"
    assert all(score.metrics["accuracy"].value == 1 for score in logs[0].results.scores)
    if benchmark == "mbpp":
        assert logs[0].results.total_samples == 5
        assert {score.reducer for score in logs[0].results.scores} >= {"pass_at_1", "pass_at_2", "pass_at_5"}
