"""The `CustomLLM` provider: bridges LiteLLM calls to a local ACP CLI agent."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import tempfile
import threading
from collections.abc import Coroutine
from typing import Any, BinaryIO

from acp import PROTOCOL_VERSION, connect_to_agent
from acp.schema import ClientCapabilities, Implementation
from litellm.llms.custom_llm import CustomLLM, CustomLLMError
from litellm.types.utils import ModelResponse

from ._version import __version__
from .adapters import AdapterSpec, resolve_and_check_adapter
from .client import ACPCollectorClient
from .errors import ACPAdapterNotFoundError
from .translate import apply_response_text, messages_to_prompt

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 300.0
STDERR_TAIL_LIMIT = 4096


class ACPCustomLLM(CustomLLM):
    """Routes `acp/<name>` completions to a local ACP CLI agent over stdio.

    Each call is stateless: it spawns a fresh adapter subprocess, opens one
    ACP session, sends the flattened conversation as a single prompt, and
    tears the subprocess down once the agent's reply is collected.
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
        timeout: float | None = None,
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
        timeout: float | None = None,
        client=None,
        **kwargs: Any,
    ) -> ModelResponse:
        assert model_response is not None  # always supplied by litellm's caller

        try:
            adapter = resolve_and_check_adapter(model)
        except ACPAdapterNotFoundError as exc:
            raise CustomLLMError(status_code=500, message=str(exc)) from exc

        prompt_timeout = timeout or DEFAULT_TIMEOUT_SECONDS
        text = await _run_acp_session(adapter, messages, prompt_timeout)
        return apply_response_text(model_response, text, model=f"acp/{model}")


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


async def _run_acp_session(adapter: AdapterSpec, messages: list, timeout: float) -> str:
    # A real file, not PIPE: a pipe's OS buffer can fill and block the agent's
    # write() once it's written enough stderr, and reading it back afterwards
    # needs every process holding the write end (grandchildren included) to
    # have closed it, which isn't guaranteed once we've merely killed `proc`.
    stderr_file = tempfile.TemporaryFile()
    try:
        proc = await asyncio.create_subprocess_exec(
            *adapter.argv(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=stderr_file,
        )
    except OSError as exc:
        stderr_file.close()
        raise CustomLLMError(
            status_code=500,
            message=f"litellm-acp: failed to start '{' '.join(adapter.argv())}': {exc}",
        ) from exc

    if proc.stdin is None or proc.stdout is None:
        proc.kill()
        stderr_file.close()
        raise CustomLLMError(
            status_code=500,
            message=f"litellm-acp: '{adapter.binary}' did not expose stdio pipes",
        )

    client_impl = ACPCollectorClient()
    conn = connect_to_agent(client_impl, proc.stdin, proc.stdout)
    try:
        await asyncio.wait_for(
            conn.initialize(
                protocol_version=PROTOCOL_VERSION,
                client_capabilities=ClientCapabilities(),
                client_info=Implementation(
                    name="litellm-acp", title="litellm-acp", version=__version__
                ),
            ),
            timeout=timeout,
        )
        session = await asyncio.wait_for(
            conn.new_session(mcp_servers=[], cwd=os.getcwd()),
            timeout=timeout,
        )
        await asyncio.wait_for(
            conn.prompt(session_id=session.session_id, prompt=messages_to_prompt(messages)),
            timeout=timeout,
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
        logger.error(
            "'%s' failed; stderr: %s",
            adapter.binary,
            _read_stderr_tail(stderr_file),
            exc_info=True,
        )
        raise
    finally:
        await _shutdown(proc, conn)
        stderr_file.close()

    return client_impl.text


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
