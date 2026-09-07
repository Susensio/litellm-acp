"""End-to-end tests: drive the real ACP wire protocol against a fake local agent.

No real subscription CLI (claude-agent-acp or codex-acp) is required -
these spawn tests/fixtures/fake_acp_agent.py, which speaks ACP for real over
stdio via the official `acp` SDK's agent-side helpers.
"""

import litellm
import pytest
from httpx import Timeout

import litellm_acp  # noqa: F401 - import side effect registers the "acp" provider
from litellm_acp import register_adapter
from litellm_acp.provider import DEFAULT_TIMEOUT_SECONDS, _resolve_timeout


def test_http_timeout_uses_its_read_deadline():
    assert _resolve_timeout(Timeout(10.0)) == 10.0
    assert _resolve_timeout(Timeout(connect=10.0, read=None, write=10.0, pool=10.0)) == (
        DEFAULT_TIMEOUT_SECONDS
    )


def test_sync_completion_round_trips_through_fake_agent(fake_agent_argv):
    binary, *args = fake_agent_argv
    register_adapter("fake-sync", binary, args=args)

    response = litellm.completion(
        model="acp/fake-sync",
        messages=[{"role": "user", "content": "hello from litellm-acp"}],
    )

    assert response.choices[0].message.content == "echo: hello from litellm-acp"
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].finish_reason == "stop"


async def test_async_completion_round_trips_through_fake_agent(fake_agent_argv):
    binary, *args = fake_agent_argv
    register_adapter("fake-async", binary, args=args)

    response = await litellm.acompletion(
        model="acp/fake-async",
        messages=[{"role": "user", "content": "async hello"}],
    )

    assert response.choices[0].message.content == "echo: async hello"


async def test_timeout_is_shared_by_the_whole_acp_session(fake_agent_argv):
    binary, *args = fake_agent_argv
    register_adapter(
        "fake-shared-timeout",
        binary,
        args=[*args, "--new-session-delay", "1.2", "--prompt-delay", "1.2"],
    )

    with pytest.raises(
        litellm.exceptions.APIConnectionError, match=r"did not respond within 2(?:\.0)?s"
    ):
        await litellm.acompletion(
            model="acp/fake-shared-timeout",
            messages=[{"role": "user", "content": "slow hello"}],
            timeout=2,
        )


def test_multi_turn_history_is_flattened_before_reaching_agent(fake_agent_argv):
    binary, *args = fake_agent_argv
    register_adapter("fake-multiturn", binary, args=args)

    response = litellm.completion(
        model="acp/fake-multiturn",
        messages=[
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "how are you"},
        ],
    )

    content = response.choices[0].message.content
    assert content.startswith("echo: ")
    assert "[User]\nhi" in content
    assert "[Assistant]\nhello" in content
