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


async def test_async_stream_delivers_chunks_as_the_agent_sends_them(fake_agent_argv):
    binary, *args = fake_agent_argv
    register_adapter("fake-stream-async", binary, args=[*args, "--chunk-delay", "0.4"])

    stream = await litellm.acompletion(
        model="acp/fake-stream-async",
        messages=[{"role": "user", "content": "one two"}],
        stream=True,
    )
    arrivals = []
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            arrivals.append((time.monotonic(), chunk.choices[0].delta.content))

    assert "".join(text for _, text in arrivals) == "echo: one two"
    assert len(arrivals) == 3
    # Two 0.4s gaps between words: buffered output would arrive all at once.
    assert arrivals[-1][0] - arrivals[0][0] > 0.5


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
