"""A minimal ACP agent used only by the litellm-acp test suite.

Spoken over stdio via the official `acp` SDK's agent-side helpers, so the
integration tests exercise the real ACP wire protocol without depending on
any real CLI (claude-agent-acp or codex-acp) being installed.
"""

import argparse
import asyncio
from typing import Any
from uuid import uuid4

from acp import Agent, InitializeResponse, NewSessionResponse, PromptResponse, run_agent
from acp.interfaces import Client
from acp.schema import (
    AgentMessageChunk,
    AudioContentBlock,
    ClientCapabilities,
    EmbeddedResourceContentBlock,
    HttpMcpServer,
    ImageContentBlock,
    Implementation,
    McpServerStdio,
    ResourceContentBlock,
    SseMcpServer,
    TextContentBlock,
)


class FakeEchoAgent(Agent):
    _conn: Client

    def __init__(self, new_session_delay: float = 0, prompt_delay: float = 0) -> None:
        self._new_session_delay = new_session_delay
        self._prompt_delay = prompt_delay

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
        text = "".join(getattr(block, "text", "") for block in prompt)
        chunk = AgentMessageChunk(
            session_update="agent_message_chunk",
            content=TextContentBlock(type="text", text=f"echo: {text}"),
        )
        await self._conn.session_update(session_id=session_id, update=chunk)
        return PromptResponse(stop_reason="end_turn")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-session-delay", type=float, default=0)
    parser.add_argument("--prompt-delay", type=float, default=0)
    args = parser.parse_args()
    await run_agent(FakeEchoAgent(args.new_session_delay, args.prompt_delay))


if __name__ == "__main__":
    asyncio.run(main())
