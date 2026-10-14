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


async def test_turn_timeout_reaches_litellm_as_its_error(fake_agent_argv):
    binary, *args = fake_agent_argv
    register_adapter("fake-timeout", binary, args=[*args, "--prompt-delay", "5"])

    with pytest.raises(
        litellm.exceptions.APIConnectionError, match=r"did not respond within 1(?:\.0)?s"
    ):
        await litellm.acompletion(
            model="acp/fake-timeout",
            messages=[{"role": "user", "content": "slow hello"}],
            timeout=1,
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


async def test_adapter_reported_usage_reaches_litellm_and_cost_is_zero(
    fake_agent_argv, logged_costs
):
    binary, *args = fake_agent_argv
    register_adapter("fake-usage", binary, args=[*args, "--report-usage"])

    response = await litellm.acompletion(
        model="acp/fake-usage",
        messages=[{"role": "user", "content": "count me"}],
    )

    usage = response.usage
    assert usage.prompt_tokens == 350  # 10 fresh + 300 cache read + 40 cache write
    assert usage.completion_tokens == 20
    assert usage.total_tokens == 370
    assert usage.prompt_tokens_details.cached_tokens == 300
    assert usage.cache_creation_input_tokens == 40
    assert usage.completion_tokens_details.reasoning_tokens == 5
    assert response._hidden_params["acp_api_equivalent_cost"] == 0.0425
    assert await logged_costs.wait_for("fake-usage") == [0.0]


async def test_adapter_without_usage_still_logs_zero_cost(fake_agent_argv, logged_costs):
    binary, *args = fake_agent_argv
    register_adapter("fake-no-usage", binary, args=args)

    response = await litellm.acompletion(
        model="acp/fake-no-usage",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert response.usage.total_tokens == 0
    assert "acp_api_equivalent_cost" not in response._hidden_params
    assert await logged_costs.wait_for("fake-no-usage") == [0.0]


def test_user_set_price_is_not_overwritten(fake_agent_argv, monkeypatch):
    binary, *args = fake_agent_argv
    register_adapter("fake-priced", binary, args=args)
    price = {"litellm_provider": "acp", "mode": "chat", "input_cost_per_token": 1.0}
    monkeypatch.setitem(litellm.model_cost, "acp/fake-priced", price)

    litellm.completion(model="acp/fake-priced", messages=[{"role": "user", "content": "hi"}])

    assert litellm.model_cost["acp/fake-priced"] is price
