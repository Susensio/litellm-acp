"""The `CustomLLM` provider: bridges LiteLLM calls to a local ACP CLI agent."""

import asyncio
import contextlib
import queue
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Coroutine, Iterator
from typing import Any, TypeVar

from httpx import Timeout
from litellm.llms.custom_llm import CustomLLM, CustomLLMError
from litellm.types.utils import GenericStreamingChunk, ModelResponse

from .adapters import AdapterSpec, resolve_and_check_adapter
from .errors import ACPAdapterNotFoundError
from .translate import fill_model_response, messages_to_prompt, price_at_zero, stream_chunk
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
        [response] = _iterate_sync(
            _yield_result(
                self.acompletion(
                    model=model,
                    messages=messages,
                    model_response=model_response,
                    timeout=timeout,
                    **kwargs,
                )
            )
        )
        return response

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
        end: TurnEnd | None = None
        async with contextlib.aclosing(_turn_events(adapter, messages, prompt_timeout)) as events:
            async for event in events:
                if isinstance(event, TextDelta):
                    parts.append(event.text)
                else:
                    end = event
        assert end is not None  # a turn that didn't raise always ends with TurnEnd
        return fill_model_response(model_response, "".join(parts), end, model=f"acp/{model}")

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
    price_at_zero(f"acp/{model}")
    return adapter


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
            yield stream_chunk(event)


def _resolve_timeout(timeout: float | Timeout | None) -> float:
    if isinstance(timeout, Timeout):
        return timeout.read or DEFAULT_TIMEOUT_SECONDS
    return timeout or DEFAULT_TIMEOUT_SECONDS


async def _yield_result(coro: Coroutine[Any, Any, T]) -> AsyncGenerator[T]:
    """A coroutine as a one-item async iterator, so `_iterate_sync` can run it too."""
    yield await coro


def _iterate_sync(agen: AsyncIterator[T]) -> Iterator[T]:
    """Drive an async iterator from sync code, on its own loop in a helper thread.

    The iterator must keep one loop for its whole life, and the caller may
    already be inside a running loop, so it can't be driven on this thread.
    The only crossing from sync into async: sync completion uses it too.
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
