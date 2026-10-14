"""Streaming (`stream=True`) against the fake ACP agent, sync and async."""

import gc
import os
import time
from pathlib import Path

import litellm
import pytest

import litellm_acp  # noqa: F401 - import side effect registers the "acp" provider
from litellm_acp import register_adapter


def _content(chunks) -> str:
    return "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)


def test_sync_stream_yields_reply_and_usage(fake_agent_argv):
    binary, *args = fake_agent_argv
    register_adapter("fake-stream-sync", binary, args=[*args, "--report-usage"])

    chunks = list(
        litellm.completion(
            model="acp/fake-stream-sync",
            messages=[{"role": "user", "content": "hello stream"}],
            stream=True,
            stream_options={"include_usage": True},
        )
    )

    assert _content(chunks) == "echo: hello stream"
    # As with OpenAI, usage follows in its own chunk after the "stop" one.
    assert "stop" in [c.choices[0].finish_reason for c in chunks if c.choices]
    assert chunks[-1].usage.prompt_tokens == 350
    assert chunks[-1].usage.completion_tokens == 20


async def test_stream_logs_zero_cost_and_carries_api_equivalent(fake_agent_argv, logged_costs):
    binary, *args = fake_agent_argv
    register_adapter("fake-stream-cost", binary, args=[*args, "--report-usage"])

    stream = await litellm.acompletion(
        model="acp/fake-stream-cost",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    fields = [c.choices[0].delta.provider_specific_fields async for c in stream if c.choices]

    assert {"acp_api_equivalent_cost": 0.0425} in fields
    assert await logged_costs.wait_for("fake-stream-cost") == [0.0]


def _wait_until_gone(pid: int, timeout: float = 10) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        # Killed but not yet reaped still counts as torn down.
        if Path(f"/proc/{pid}/stat").read_text().split()[2] == "Z":
            return True
        time.sleep(0.1)
    return False


@pytest.mark.skipif(not Path("/proc").exists(), reason="checks process state via /proc")
def test_abandoned_sync_stream_tears_the_adapter_down(fake_agent_argv, tmp_path):
    binary, *args = fake_agent_argv
    pid_file = tmp_path / "pid"
    register_adapter(
        "fake-stream-abandon",
        binary,
        args=[*args, "--chunk-delay", "30", "--pid-file", str(pid_file)],
    )

    stream = litellm.completion(
        model="acp/fake-stream-abandon",
        messages=[{"role": "user", "content": "never finishes"}],
        stream=True,
    )
    first = next(iter(stream))
    assert first.choices[0].delta.content == "echo:"
    del stream, first
    gc.collect()

    assert _wait_until_gone(int(pid_file.read_text()))


async def test_unknown_adapter_fails_when_the_stream_is_requested():
    with pytest.raises(litellm.exceptions.APIConnectionError, match="no ACP adapter"):
        await litellm.acompletion(
            model="acp/not-registered-for-streaming",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
