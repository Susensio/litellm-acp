"""One ACP turn: spawn an adapter, send one prompt, stream back what the agent says.

`run_turn` is the whole interface. Everything about the adapter subprocess -
spawning it, the ACP handshake, the shared deadline, stderr capture, and
tearing it down however the turn ends - stays in this module.

The ACP `Client` here is headless: litellm-acp advertises no filesystem or
terminal capabilities during `initialize()`, so a well-behaved agent shouldn't
ask for them. Everything other than `session_update` is implemented
defensively in case an agent asks anyway.
"""

import asyncio
import contextlib
import logging
import os
import tempfile
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any, BinaryIO

from acp import PROTOCOL_VERSION, Client, RequestError, connect_to_agent
from acp.schema import (
    AgentMessageChunk,
    AgentPlanContentUpdate,
    AgentPlanRemovedUpdate,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AudioContentBlock,
    AvailableCommandsUpdate,
    ClientCapabilities,
    ConfigOptionUpdate,
    Cost,
    CreateElicitationResponse,
    CreateTerminalResponse,
    CurrentModeUpdate,
    DeclineElicitationResponse,
    DeniedOutcome,
    ElicitationMode,
    EmbeddedResourceContentBlock,
    EnvVariable,
    ImageContentBlock,
    Implementation,
    KillTerminalResponse,
    PermissionOption,
    ReadTextFileResponse,
    ReleaseTerminalResponse,
    RequestPermissionResponse,
    ResourceContentBlock,
    SessionInfoUpdate,
    TerminalOutputResponse,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    ToolCallUpdate,
    Usage,
    UsageUpdate,
    UserMessageChunk,
    WaitForTerminalExitResponse,
    WriteTextFileResponse,
)

from .adapters import AdapterSpec
from .translate import ACPContentBlock

logger = logging.getLogger(__name__)

STDERR_TAIL_LIMIT = 4096


@dataclass(frozen=True)
class TextDelta:
    """A piece of the agent's reply, in the order the agent sent it."""

    text: str


@dataclass(frozen=True)
class TurnEnd:
    """The last event of a turn that completed."""

    usage: Usage | None
    # Cumulative for the session; each turn runs in a fresh session, so this is the turn's.
    cost: Cost | None
    stop_reason: str


TurnEvent = TextDelta | TurnEnd


class TurnError(Exception):
    """A turn that didn't complete. `stderr` is the adapter's last output, when there was one."""

    def __init__(self, message: str, stderr: str | None = None) -> None:
        super().__init__(message)
        self.stderr = stderr


class TurnTimeoutError(TurnError):
    """The adapter didn't start or answer before the turn's deadline."""


class TurnFailedError(TurnError):
    """The adapter couldn't be started."""


async def run_turn(
    adapter: AdapterSpec, prompt: list[ACPContentBlock], timeout: float
) -> AsyncGenerator[TurnEvent]:
    """Run one prompt in a fresh adapter subprocess: `TextDelta`s, then one `TurnEnd`.

    `timeout` covers the whole turn, from spawning to the last reply. Raises
    `TurnError`s for a turn that couldn't run; errors the agent itself reports
    propagate as the ACP SDK raised them. Closing the iterator early tears the
    adapter down.
    """
    # None marks the end of the turn, successful or not.
    texts: asyncio.Queue[str | None] = asyncio.Queue()
    turn = asyncio.create_task(_run(adapter, prompt, timeout, texts.put_nowait))
    turn.add_done_callback(lambda _: texts.put_nowait(None))
    try:
        while (text := await texts.get()) is not None:
            yield TextDelta(text)
        yield await turn
    finally:
        # Only still running if the caller stopped consuming early.
        turn.cancel()
        await asyncio.gather(turn, return_exceptions=True)


async def _run(
    adapter: AdapterSpec,
    prompt: list[ACPContentBlock],
    timeout: float,
    on_text: Callable[[str], None],
) -> TurnEnd:
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
            raise TurnTimeoutError(f"litellm-acp: timed out starting '{adapter.binary}'.") from exc
        except OSError as exc:
            raise TurnFailedError(
                f"litellm-acp: failed to start '{' '.join(adapter.argv())}': {exc}"
            ) from exc

        if proc.stdin is None or proc.stdout is None:
            proc.kill()
            raise TurnFailedError(f"litellm-acp: '{adapter.binary}' did not expose stdio pipes")

        client = _Client(on_text)
        conn = connect_to_agent(client, proc.stdin, proc.stdout)
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
            response = await asyncio.wait_for(
                conn.prompt(session_id=session.session_id, prompt=prompt),
                timeout=_remaining_timeout(deadline),
            )
        except TimeoutError as exc:
            stderr = _read_stderr_tail(stderr_file)
            logger.error(
                "'%s' did not respond within %ss; stderr: %s", adapter.binary, timeout, stderr
            )
            raise TurnTimeoutError(
                f"litellm-acp: '{adapter.binary}' did not respond within {timeout}s. "
                "Pass a longer timeout=... to litellm.completion(...) if the agent is just slow.",
                stderr=stderr,
            ) from exc
        except Exception:
            logger.exception(
                "'%s' failed; stderr: %s", adapter.binary, _read_stderr_tail(stderr_file)
            )
            raise
        finally:
            await _shutdown(proc, conn)

        return TurnEnd(usage=response.usage, cost=client.cost, stop_reason=response.stop_reason)


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


class _Client(Client):
    """Passes `AgentMessageChunk` text on as it arrives and keeps the latest `UsageUpdate` cost."""

    def __init__(self, on_text: Callable[[str], None]) -> None:
        self._on_text = on_text
        self.cost: Cost | None = None

    async def session_update(
        self,
        session_id: str,
        update: UserMessageChunk
        | AgentMessageChunk
        | AgentThoughtChunk
        | ToolCallStart
        | ToolCallProgress
        | AgentPlanUpdate
        | AgentPlanContentUpdate
        | AgentPlanRemovedUpdate
        | AvailableCommandsUpdate
        | CurrentModeUpdate
        | ConfigOptionUpdate
        | SessionInfoUpdate
        | UsageUpdate,
        **kwargs: Any,
    ) -> None:
        if isinstance(update, AgentMessageChunk):
            if text := _text_of(update.content):
                self._on_text(text)
        elif isinstance(update, UsageUpdate) and update.cost is not None:
            self.cost = update.cost

    async def request_permission(
        self,
        session_id: str,
        tool_call: ToolCallUpdate,
        options: list[PermissionOption],
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    async def write_text_file(
        self, session_id: str, path: str, content: str, **kwargs: Any
    ) -> WriteTextFileResponse | None:
        raise RequestError.method_not_found("fs/write_text_file")

    async def read_text_file(
        self,
        session_id: str,
        path: str,
        line: int | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> ReadTextFileResponse:
        raise RequestError.method_not_found("fs/read_text_file")

    async def create_terminal(
        self,
        session_id: str,
        command: str,
        args: list[str] | None = None,
        env: list[EnvVariable] | None = None,
        cwd: str | None = None,
        output_byte_limit: int | None = None,
        **kwargs: Any,
    ) -> CreateTerminalResponse:
        raise RequestError.method_not_found("terminal/create")

    async def terminal_output(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> TerminalOutputResponse:
        raise RequestError.method_not_found("terminal/output")

    async def release_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> ReleaseTerminalResponse | None:
        raise RequestError.method_not_found("terminal/release")

    async def wait_for_terminal_exit(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> WaitForTerminalExitResponse:
        raise RequestError.method_not_found("terminal/wait_for_exit")

    async def kill_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> KillTerminalResponse | None:
        raise RequestError.method_not_found("terminal/kill")

    async def create_elicitation(
        self, message: str, mode: ElicitationMode, **kwargs: Any
    ) -> CreateElicitationResponse:
        return DeclineElicitationResponse(action="decline")

    async def complete_elicitation(self, elicitation_id: str, **kwargs: Any) -> None:
        return None

    async def ext_method(self, method: str, params: dict) -> dict:
        raise RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: dict) -> None:
        return None


def _text_of(content: object) -> str:
    if isinstance(content, TextContentBlock):
        return content.text
    if isinstance(content, ImageContentBlock):
        return "<image>"
    if isinstance(content, AudioContentBlock):
        return "<audio>"
    if isinstance(content, ResourceContentBlock):
        return content.uri or "<resource>"
    if isinstance(content, EmbeddedResourceContentBlock):
        return "<resource>"
    return ""
