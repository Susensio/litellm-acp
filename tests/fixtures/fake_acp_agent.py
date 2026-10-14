"""A minimal ACP agent used only by the litellm-acp test suite.

Spoken over stdio via the official `acp` SDK's agent-side helpers, so the
integration tests exercise the real ACP wire protocol without depending on
any real CLI (claude-agent-acp or codex-acp) being installed.
"""

import argparse
import asyncio
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from acp import Agent, InitializeResponse, NewSessionResponse, PromptResponse, run_agent
from acp.interfaces import Client
from acp.schema import (
    AgentMessageChunk,
    AudioContentBlock,
    ClientCapabilities,
    Cost,
    EmbeddedResourceContentBlock,
    HttpMcpServer,
    ImageContentBlock,
    Implementation,
    McpServerStdio,
    ResourceContentBlock,
    SseMcpServer,
    TextContentBlock,
    Usage,
    UsageUpdate,
)


class FakeEchoAgent(Agent):
    _conn: Client

    def __init__(
        self,
        new_session_delay: float = 0,
        prompt_delay: float = 0,
        report_usage: bool = False,
        chunk_delay: float | None = None,
    ) -> None:
        self._new_session_delay = new_session_delay
        self._prompt_delay = prompt_delay
        self._report_usage = report_usage
        # When set, the echo is sent word by word with this delay between words.
        self._chunk_delay = chunk_delay

    def on_connect(self, conn: Client) -> None:
        self._conn = conn

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        return InitializeResponse(protocol_version=protocol_version)

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        await asyncio.sleep(self._new_session_delay)
        return NewSessionResponse(session_id=uuid4().hex)

    async def prompt(
        self,
        session_id: str,
        prompt: list[
            TextContentBlock
            | ImageContentBlock
            | AudioContentBlock
            | ResourceContentBlock
            | EmbeddedResourceContentBlock
        ],
        **kwargs: Any,
    ) -> PromptResponse:
        await asyncio.sleep(self._prompt_delay)
        text = "echo: " + "".join(getattr(block, "text", "") for block in prompt)
        pieces = [text] if self._chunk_delay is None else text.split(" ")
        for i, piece in enumerate(pieces):
            if i:
                await asyncio.sleep(self._chunk_delay or 0)
                piece = " " + piece
            chunk = AgentMessageChunk(
                session_update="agent_message_chunk",
                content=TextContentBlock(type="text", text=piece),
            )
            await self._conn.session_update(session_id=session_id, update=chunk)
        if not self._report_usage:
            return PromptResponse(stop_reason="end_turn")

        # Shaped like claude-agent-acp: input_tokens net of the cache counters,
        # cost as a cumulative usage_update (sent twice: only the last counts).
        for amount in (0.01, 0.0425):
            await self._conn.session_update(
                session_id=session_id,
                update=UsageUpdate(
                    session_update="usage_update",
                    used=1000,
                    size=200000,
                    cost=Cost(amount=amount, currency="USD"),
                ),
            )
        usage = Usage(
            input_tokens=10,
            output_tokens=20,
            cached_read_tokens=300,
            cached_write_tokens=40,
            thought_tokens=5,
            total_tokens=370,
        )
        return PromptResponse(stop_reason="end_turn", usage=usage)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-session-delay", type=float, default=0)
    parser.add_argument("--prompt-delay", type=float, default=0)
    parser.add_argument("--report-usage", action="store_true")
    parser.add_argument("--chunk-delay", type=float)
    parser.add_argument("--pid-file")
    args = parser.parse_args()
    if args.pid_file:
        Path(args.pid_file).write_text(str(os.getpid()))
    await run_agent(
        FakeEchoAgent(
            args.new_session_delay, args.prompt_delay, args.report_usage, args.chunk_delay
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
