"""`run_turn` against the fake ACP agent, without LiteLLM in the way."""

import asyncio
import contextlib
import os
import time
from pathlib import Path

import pytest
from acp.schema import TextContentBlock

from litellm_acp.adapters import AdapterSpec
from litellm_acp.turn import TextDelta, TurnEnd, TurnFailedError, TurnTimeoutError, run_turn


def _prompt(text: str) -> list:
    return [TextContentBlock(type="text", text=text)]


def _fake(fake_agent_argv, *extra: str) -> AdapterSpec:
    binary, *args = fake_agent_argv
    return AdapterSpec(binary, [*args, *extra])


async def _collect(adapter: AdapterSpec, text: str, timeout: float = 30) -> list:
    return [event async for event in run_turn(adapter, _prompt(text), timeout)]


async def test_text_deltas_arrive_in_order_then_turn_end(fake_agent_argv):
    events = await _collect(_fake(fake_agent_argv, "--chunk-delay", "0"), "one two")

    assert events[:-1] == [TextDelta("echo:"), TextDelta(" one"), TextDelta(" two")]
    assert events[-1] == TurnEnd(usage=None, cost=None, stop_reason="end_turn")


async def test_turn_end_carries_reported_usage_and_last_cost(fake_agent_argv):
    *_, end = await _collect(_fake(fake_agent_argv, "--report-usage"), "hi")

    assert end.usage.input_tokens == 10
    assert end.usage.cached_read_tokens == 300
    assert end.cost.amount == 0.0425  # the last of the cumulative usage_updates


async def test_deltas_are_yielded_as_the_agent_sends_them(fake_agent_argv):
    arrivals = []
    async for event in run_turn(
        _fake(fake_agent_argv, "--chunk-delay", "0.4"), _prompt("one two"), 30
    ):
        if isinstance(event, TextDelta):
            arrivals.append(time.monotonic())

    assert len(arrivals) == 3
    # Two 0.4s gaps between words: buffered output would arrive all at once.
    assert arrivals[-1] - arrivals[0] > 0.5


async def test_one_deadline_covers_the_whole_turn(fake_agent_argv):
    # Each step alone fits in the deadline; together they don't.
    adapter = _fake(fake_agent_argv, "--new-session-delay", "0.6", "--prompt-delay", "0.6")

    with pytest.raises(TurnTimeoutError, match=r"did not respond within 1(?:\.0)?s") as excinfo:
        await _collect(adapter, "slow", timeout=1)
    assert excinfo.value.stderr is not None


async def test_unstartable_adapter_fails_the_turn():
    with pytest.raises(TurnFailedError, match="failed to start"):
        await _collect(AdapterSpec("/nonexistent/acp-adapter"), "hi")


@pytest.mark.skipif(not Path("/proc").exists(), reason="checks process state via /proc")
async def test_closing_the_turn_early_tears_the_adapter_down(fake_agent_argv, tmp_path):
    pid_file = tmp_path / "pid"
    adapter = _fake(fake_agent_argv, "--chunk-delay", "30", "--pid-file", str(pid_file))

    async with contextlib.aclosing(run_turn(adapter, _prompt("never finishes"), 60)) as events:
        assert await anext(events) == TextDelta("echo:")

    assert await _gone(int(pid_file.read_text()))


async def _gone(pid: int, timeout: float = 10) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        # Killed but not yet reaped still counts as torn down.
        if Path(f"/proc/{pid}/stat").read_text().split()[2] == "Z":
            return True
        await asyncio.sleep(0.1)
    return False
