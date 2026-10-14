"""Translate between LiteLLM's OpenAI shapes and ACP turns, in both directions.

In: OpenAI messages become one ACP prompt. Out: a turn's events become a
`ModelResponse` or streaming chunks, including the cost policy - LiteLLM sees
$0 for `acp/*` calls, and the adapter's API-price estimate rides alongside.
"""

from typing import Literal

import litellm
from acp.schema import Cost, TextContentBlock, Usage
from litellm.types.llms.openai import ChatCompletionUsageBlock
from litellm.types.utils import GenericStreamingChunk, ModelResponse
from litellm.types.utils import Usage as LiteLLMUsage

from .turn import ACPContentBlock, TextDelta, TurnEnd, TurnEvent

_ROLE_LABELS = {"system": "System", "user": "User", "assistant": "Assistant", "tool": "Tool"}


def _content_to_text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
                else:
                    parts.append(f"<{part.get('type', 'content')}>")
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)


def messages_to_prompt(messages: list[dict]) -> list[ACPContentBlock]:
    """Flatten an OpenAI-style message list into a single ACP text prompt block.

    ACP agent sessions don't accept resent OpenAI-style history the way a
    stateless chat-completions API does, so each litellm-acp call starts a
    fresh ACP session and renders the full transcript as one labeled text
    block. A single trailing user turn is sent as-is, without labels.
    """
    if not messages:
        return [TextContentBlock(type="text", text="")]

    if len(messages) == 1 and messages[0].get("role") == "user":
        return [TextContentBlock(type="text", text=_content_to_text(messages[0].get("content")))]

    lines = []
    for message in messages:
        role = message.get("role", "user")
        label = _ROLE_LABELS.get(role, role)
        lines.append(f"[{label}]\n{_content_to_text(message.get('content'))}")
    return [TextContentBlock(type="text", text="\n\n".join(lines))]


API_EQUIVALENT_COST_KEY = "acp_api_equivalent_cost"


def fill_model_response(
    model_response: ModelResponse, text: str, end: TurnEnd, model: str
) -> ModelResponse:
    """Populate litellm's pre-built ModelResponse in place with a finished turn.

    Whatever the adapter didn't report (usage, cost) is left at LiteLLM's defaults.
    """
    choice = model_response.choices[0]
    choice.message.content = text
    choice.message.role = "assistant"
    choice.finish_reason = _finish_reason(end.stop_reason)
    model_response.model = model
    if end.usage is not None:
        # `usage` isn't a declared field of ModelResponse; litellm sets it the same way.
        setattr(model_response, "usage", _to_litellm_usage(end.usage))  # noqa: B010
    if (amount := _api_equivalent_cost(end.cost)) is not None:
        model_response._hidden_params[API_EQUIVALENT_COST_KEY] = amount
    return model_response


def stream_chunk(event: TurnEvent) -> GenericStreamingChunk:
    """One turn event as a LiteLLM streaming chunk; the `TurnEnd` one finishes the stream."""
    if isinstance(event, TextDelta):
        return _chunk(event.text)
    full = _to_litellm_usage(event.usage) if event.usage is not None else None
    chunk = _chunk(
        "",
        finish_reason=_finish_reason(event.stop_reason),
        usage=ChatCompletionUsageBlock(
            prompt_tokens=full.prompt_tokens,
            completion_tokens=full.completion_tokens,
            total_tokens=full.total_tokens,
            prompt_tokens_details=full.prompt_tokens_details.model_dump()
            if full.prompt_tokens_details
            else None,
            completion_tokens_details=full.completion_tokens_details.model_dump()
            if full.completion_tokens_details
            else None,
        )
        if full is not None
        else None,
    )
    if (amount := _api_equivalent_cost(event.cost)) is not None:
        chunk["provider_specific_fields"] = {API_EQUIVALENT_COST_KEY: amount}
    return chunk


def price_at_zero(model: str) -> None:
    """Give `model` a zero price in LiteLLM's cost map, unless the user priced it already.

    A subscription call costs nothing at the margin, so cost tracking and
    budgets should see $0 - not "unmapped model", which logs no cost at all.
    Written directly rather than via `litellm.register_model`, which prints
    lookup errors to stdout for providers it doesn't know.
    """
    litellm.model_cost.setdefault(
        model,
        {
            "litellm_provider": "acp",
            "mode": "chat",
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "cache_creation_input_token_cost": 0.0,
            "cache_read_input_token_cost": 0.0,
        },
    )


def _chunk(
    text: str, finish_reason: str = "", usage: ChatCompletionUsageBlock | None = None
) -> GenericStreamingChunk:
    return GenericStreamingChunk(
        text=text,
        tool_use=None,
        is_finished=bool(finish_reason),
        finish_reason=finish_reason,
        usage=usage,
        index=0,
    )


# ACP StopReason -> OpenAI finish_reason. "cancelled" only follows a client-side
# cancel, which litellm-acp never sends, so it falls through to "stop".
_FinishReason = Literal["stop", "length", "content_filter"]
_FINISH_REASONS: dict[str, _FinishReason] = {
    "max_tokens": "length",
    "max_turn_requests": "length",
    "refusal": "content_filter",
}


def _finish_reason(stop_reason: str) -> _FinishReason:
    return _FINISH_REASONS.get(stop_reason, "stop")


def _to_litellm_usage(usage: Usage) -> LiteLLMUsage:
    """Convert an ACP turn's token usage into LiteLLM's OpenAI-shaped `Usage`.

    ACP adapters report `input_tokens` net of cache reads and writes (Anthropic
    semantics, which claude-agent-acp and codex-acp both normalize to), whereas
    OpenAI's `prompt_tokens` includes them - so the cache counters are added back.
    """
    cache_read = usage.cached_read_tokens or 0
    cache_write = usage.cached_write_tokens or 0
    prompt_tokens = usage.input_tokens + cache_read + cache_write
    return LiteLLMUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=usage.output_tokens,
        total_tokens=prompt_tokens + usage.output_tokens,
        prompt_tokens_details={"cached_tokens": cache_read},
        completion_tokens_details={"reasoning_tokens": usage.thought_tokens},
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
    )


def _api_equivalent_cost(cost: Cost | None) -> float | None:
    """The adapter's estimate of what the call would cost at API prices, in USD.

    Not what the subscription charges - informational only, never `response_cost`.
    """
    if cost is None or cost.currency.upper() != "USD":
        return None
    return cost.amount
