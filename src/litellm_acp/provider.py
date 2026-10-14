"""The `CustomLLM` provider: bridges LiteLLM calls to a local ACP CLI agent."""

import asyncio
import contextlib
import queue
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Coroutine, Iterator
from typing import Any, TypeVar

import litellm
from acp.schema import Usage
from httpx import Timeout
from litellm.llms.custom_llm import CustomLLM, CustomLLMError
from litellm.types.llms.openai import ChatCompletionUsageBlock
from litellm.types.utils import GenericStreamingChunk, ModelResponse

from .adapters import AdapterSpec, resolve_and_check_adapter
from .errors import ACPAdapterNotFoundError
from .translate import (
    API_EQUIVALENT_COST_KEY,
    api_equivalent_cost,
    apply_response_text,
    apply_usage,
    messages_to_prompt,
    to_litellm_usage,
)
from .turn import TextDelta, TurnEnd, TurnEvent, TurnFailedError, TurnTimeoutError, run_turn

DEFAULT_TIMEOUT_SECONDS = 300.0

T = TypeVar("T")


class ACPCustomLLM(CustomLLM):
    """Routes `acp/<name>` completions to a local ACP CLI agent over stdio.

    Each call is one stateless ACP turn (see `turn.run_turn`) carrying the
    flattened conversation as a single prompt. Completion collects the turn;
    streaming yields each agent message chunk as it arrives.
    """

    def completion(
        self,
        model: str,
        messages: list,
        api_base=None,
        custom_prompt_dict: dict | None = None,
        model_response: ModelResponse | None = None,
        print_verbose=None,
        encoding=None,
        api_key=None,
        logging_obj=None,
        optional_params: dict | None = None,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers: dict | None = None,
        timeout=None,
        client=None,
        **kwargs: Any,
    ) -> ModelResponse:
        return _run_sync(
            self.acompletion(
                model=model,
                messages=messages,
                model_response=model_response,
                timeout=timeout,
                **kwargs,
            )
        )

    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base=None,
        custom_prompt_dict: dict | None = None,
        model_response: ModelResponse | None = None,
        print_verbose=None,
        encoding=None,
        api_key=None,
        logging_obj=None,
        optional_params: dict | None = None,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers: dict | None = None,
        timeout=None,
        client=None,
        **kwargs: Any,
    ) -> ModelResponse:
        assert model_response is not None  # always supplied by litellm's caller

        adapter = _resolve_adapter(model)

        prompt_timeout = _resolve_timeout(timeout)
        parts: list[str] = []
        async with contextlib.aclosing(_turn_events(adapter, messages, prompt_timeout)) as events:
            async for event in events:
                if isinstance(event, TextDelta):
                    parts.append(event.text)
                else:
                    apply_usage(model_response, event.usage, event.cost)
        return apply_response_text(model_response, "".join(parts), model=f"acp/{model}")

    def streaming(
        self,
        model: str,
        messages: list,
        api_base=None,
        custom_prompt_dict: dict | None = None,
        model_response: ModelResponse | None = None,
        print_verbose=None,
        encoding=None,
        api_key=None,
        logging_obj=None,
        optional_params: dict | None = None,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers: dict | None = None,
        timeout=None,
        client=None,
        **kwargs: Any,
    ) -> Iterator[GenericStreamingChunk]:
        adapter = _resolve_adapter(model)
        return _iterate_sync(_stream(adapter, messages, _resolve_timeout(timeout)))

    # Not a coroutine, unlike the base class: litellm iterates the return value
    # directly without awaiting it, and a plain def lets an unknown adapter fail
    # at call time rather than on the first chunk.
    def astreaming(  # ty: ignore[invalid-method-override]
        self,
        model: str,
        messages: list,
        api_base=None,
        custom_prompt_dict: dict | None = None,
        model_response: ModelResponse | None = None,
        print_verbose=None,
        encoding=None,
        api_key=None,
        logging_obj=None,
        optional_params: dict | None = None,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers: dict | None = None,
        timeout=None,
        client=None,
        **kwargs: Any,
    ) -> AsyncIterator[GenericStreamingChunk]:
        adapter = _resolve_adapter(model)
        return _stream(adapter, messages, _resolve_timeout(timeout))


def _resolve_adapter(model: str) -> AdapterSpec:
    try:
        adapter = resolve_and_check_adapter(model)
    except ACPAdapterNotFoundError as exc:
        raise CustomLLMError(status_code=500, message=str(exc)) from exc
    _price_at_zero(f"acp/{model}")
    return adapter


def _price_at_zero(model: str) -> None:
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


async def _turn_events(
    adapter: AdapterSpec, messages: list, timeout: float
) -> AsyncGenerator[TurnEvent]:
    """`run_turn` for a LiteLLM call: OpenAI messages in, turn errors mapped to LiteLLM's."""
    try:
        async with contextlib.aclosing(
            run_turn(adapter, messages_to_prompt(messages), timeout)
        ) as events:
            async for event in events:
                yield event
    except TurnTimeoutError as exc:
        raise CustomLLMError(status_code=504, message=str(exc)) from exc
    except TurnFailedError as exc:
        raise CustomLLMError(status_code=500, message=str(exc)) from exc


async def _stream(
    adapter: AdapterSpec, messages: list, timeout: float
) -> AsyncGenerator[GenericStreamingChunk]:
    async with contextlib.aclosing(_turn_events(adapter, messages, timeout)) as events:
        async for event in events:
            if isinstance(event, TextDelta):
                yield _stream_chunk(event.text)
            else:
                yield _end_chunk(event)


def _end_chunk(end: TurnEnd) -> GenericStreamingChunk:
    return _stream_chunk(
        "",
        finish_reason="stop",
        usage=_stream_usage(end.usage) if end.usage is not None else None,
        api_cost=api_equivalent_cost(end.cost),
    )


def _stream_usage(usage: Usage) -> ChatCompletionUsageBlock:
    full = to_litellm_usage(usage)
    return ChatCompletionUsageBlock(
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


def _stream_chunk(
    text: str,
    finish_reason: str = "",
    usage: ChatCompletionUsageBlock | None = None,
    api_cost: float | None = None,
) -> GenericStreamingChunk:
    chunk = GenericStreamingChunk(
        text=text,
        tool_use=None,
        is_finished=bool(finish_reason),
        finish_reason=finish_reason,
        usage=usage,
        index=0,
    )
    if api_cost is not None:
        chunk["provider_specific_fields"] = {API_EQUIVALENT_COST_KEY: api_cost}
    return chunk


def _resolve_timeout(timeout: float | Timeout | None) -> float:
    if isinstance(timeout, Timeout):
        return timeout.read or DEFAULT_TIMEOUT_SECONDS
    return timeout or DEFAULT_TIMEOUT_SECONDS


def _run_sync(coro: Coroutine[Any, Any, ModelResponse]) -> ModelResponse:
    """Run an async completion from sync code, whether or not a loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # We're already inside a running event loop (e.g. an async caller invoked
    # the sync `completion()` from a sync-wrapping context) - asyncio.run()
    # would raise here, so drive the coroutine on its own loop in a thread.
    outcome: dict[str, Any] = {}

    def _runner() -> None:
        try:
            outcome["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread below
            outcome["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()

    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def _iterate_sync(agen: AsyncIterator[T]) -> Iterator[T]:
    """Drive an async iterator from sync code, on its own loop in a helper thread.

    The iterator must keep one loop for its whole life, and - as in `_run_sync` -
    the caller may already be inside a running loop, so it can't be driven here.
    """
    items: queue.Queue[tuple[str, Any]] = queue.Queue()
    started = threading.Event()
    pump_state: dict[str, Any] = {}

    async def _pump() -> None:
        pump_state["loop"] = asyncio.get_running_loop()
        pump_state["task"] = asyncio.current_task()
        started.set()
        try:
            async for item in agen:
                items.put(("item", item))
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread below
            items.put(("error", exc))
        else:
            items.put(("done", None))

    thread = threading.Thread(target=asyncio.run, args=(_pump(),), daemon=True)
    thread.start()
    try:
        while True:
            kind, value = items.get()
            if kind == "item":
                yield value
            elif kind == "error":
                raise value
            else:
                return
    finally:
        # Only does anything if the caller stopped consuming early.
        started.wait()
        with contextlib.suppress(RuntimeError):  # loop already closed: pump finished
            pump_state["loop"].call_soon_threadsafe(pump_state["task"].cancel)
        thread.join()
