"""The `CustomLLM` provider: bridges LiteLLM calls to a local ACP CLI agent."""

import asyncio
import contextlib
import logging
import os
import queue
import tempfile
import threading
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator
from importlib.metadata import version
from typing import Any, BinaryIO, TypeVar

from acp import PROTOCOL_VERSION, connect_to_agent
from acp.schema import ClientCapabilities, Cost, Implementation, Usage
from httpx import Timeout
from litellm.llms.custom_llm import CustomLLM, CustomLLMError
from litellm.types.llms.openai import ChatCompletionUsageBlock
from litellm.types.utils import GenericStreamingChunk, ModelResponse

from .adapters import AdapterSpec, resolve_and_check_adapter
from .client import ACPCollectorClient
from .errors import ACPAdapterNotFoundError
from .translate import apply_response_text, apply_usage, messages_to_prompt, to_litellm_usage

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 300.0
STDERR_TAIL_LIMIT = 4096

T = TypeVar("T")


class ACPCustomLLM(CustomLLM):
    """Routes `acp/<name>` completions to a local ACP CLI agent over stdio.

    Each call is stateless: it spawns a fresh adapter subprocess, opens one
    ACP session, sends the flattened conversation as a single prompt, and
    tears the subprocess down once the agent's reply is collected. Streaming
    calls do the same, yielding each agent message chunk as it arrives.
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
        text, usage, cost = await _run_acp_session(adapter, messages, prompt_timeout)
        apply_usage(model_response, usage, cost)
        return apply_response_text(model_response, text, model=f"acp/{model}")

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
        return _iterate_sync(_stream_acp_session(adapter, messages, _resolve_timeout(timeout)))

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
        return _stream_acp_session(adapter, messages, _resolve_timeout(timeout))


def _resolve_adapter(model: str) -> AdapterSpec:
    try:
        return resolve_and_check_adapter(model)
    except ACPAdapterNotFoundError as exc:
        raise CustomLLMError(status_code=500, message=str(exc)) from exc


async def _stream_acp_session(
    adapter: AdapterSpec, messages: list, timeout: float
) -> AsyncIterator[GenericStreamingChunk]:
    # None marks the end of the session, successful or not.
    texts: asyncio.Queue[str | None] = asyncio.Queue()
    session = asyncio.create_task(
        _run_acp_session(adapter, messages, timeout, on_text=texts.put_nowait)
    )
    session.add_done_callback(lambda _: texts.put_nowait(None))
    try:
        while (text := await texts.get()) is not None:
            yield _stream_chunk(text)
        _, usage, _ = await session
        yield _stream_chunk(
            "",
            finish_reason="stop",
            usage=_stream_usage(usage) if usage is not None else None,
        )
    finally:
        # Only still running if the caller stopped consuming early: tear the
        # adapter down rather than leave it generating into the void.
        session.cancel()
        await asyncio.gather(session, return_exceptions=True)


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


async def _run_acp_session(
    adapter: AdapterSpec,
    messages: list,
    timeout: float,
    on_text: Callable[[str], None] | None = None,
) -> tuple[str, Usage | None, Cost | None]:
    # A real file, not PIPE: a pipe's OS buffer can fill and block the agent's
    # write() once it's written enough stderr, and reading it back afterwards
    # needs every process holding the write end (grandchildren included) to
    # have closed it, which isn't guaranteed once we've merely killed `proc`.
    deadline = asyncio.get_running_loop().time() + timeout
    with tempfile.TemporaryFile() as stderr_file:
        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *adapter.argv(),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=stderr_file,
                ),
                timeout=_remaining_timeout(deadline),
            )
        except TimeoutError as exc:
            raise CustomLLMError(
                status_code=504,
                message=f"litellm-acp: timed out starting '{adapter.binary}'.",
            ) from exc
        except OSError as exc:
            raise CustomLLMError(
                status_code=500,
                message=f"litellm-acp: failed to start '{' '.join(adapter.argv())}': {exc}",
            ) from exc

        if proc.stdin is None or proc.stdout is None:
            proc.kill()
            raise CustomLLMError(
                status_code=500,
                message=f"litellm-acp: '{adapter.binary}' did not expose stdio pipes",
            )

        client_impl = ACPCollectorClient(on_text=on_text)
        conn = connect_to_agent(client_impl, proc.stdin, proc.stdout)
        try:
            await asyncio.wait_for(
                conn.initialize(
                    protocol_version=PROTOCOL_VERSION,
                    client_capabilities=ClientCapabilities(),
                    client_info=Implementation(
                        name="litellm-acp", title="litellm-acp", version=version("litellm-acp")
                    ),
                ),
                timeout=_remaining_timeout(deadline),
            )
            session = await asyncio.wait_for(
                conn.new_session(mcp_servers=[], cwd=os.getcwd()),
                timeout=_remaining_timeout(deadline),
            )
            prompt_response = await asyncio.wait_for(
                conn.prompt(session_id=session.session_id, prompt=messages_to_prompt(messages)),
                timeout=_remaining_timeout(deadline),
            )
        except TimeoutError as exc:
            logger.error(
                "'%s' did not respond within %ss; stderr: %s",
                adapter.binary,
                timeout,
                _read_stderr_tail(stderr_file),
            )
            raise CustomLLMError(
                status_code=504,
                message=(
                    f"litellm-acp: '{adapter.binary}' did not respond within {timeout}s. "
                    "Pass a longer timeout=... to litellm.completion(...) if the agent is just slow."
                ),
            ) from exc
        except Exception:
            logger.exception(
                "'%s' failed; stderr: %s",
                adapter.binary,
                _read_stderr_tail(stderr_file),
            )
            raise
        finally:
            await _shutdown(proc, conn)

        return client_impl.text, prompt_response.usage, client_impl.cost


def _remaining_timeout(deadline: float) -> float:
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        raise TimeoutError
    return remaining


def _read_stderr_tail(stderr_file: BinaryIO) -> str:
    stderr_file.seek(0)
    data = stderr_file.read()
    return data.decode(errors="replace").strip()[-STDERR_TAIL_LIMIT:] or "<empty>"


async def _shutdown(proc: asyncio.subprocess.Process, conn: Any) -> None:
    with contextlib.suppress(Exception):
        await asyncio.wait_for(conn.close(), timeout=2)
    if proc.stdin is not None:
        with contextlib.suppress(Exception):
            proc.stdin.close()
            await asyncio.wait_for(proc.stdin.wait_closed(), timeout=2)
    if proc.returncode is None:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except TimeoutError:
            proc.kill()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(proc.wait(), timeout=5)
