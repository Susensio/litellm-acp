"""Translate between LiteLLM's OpenAI-shaped messages and the ACP protocol."""

from acp.schema import (
    AudioContentBlock,
    EmbeddedResourceContentBlock,
    ImageContentBlock,
    ResourceContentBlock,
    TextContentBlock,
)
from litellm.types.utils import ModelResponse

_ROLE_LABELS = {"system": "System", "user": "User", "assistant": "Assistant", "tool": "Tool"}
ACPContentBlock = (
    TextContentBlock
    | ImageContentBlock
    | AudioContentBlock
    | ResourceContentBlock
    | EmbeddedResourceContentBlock
)


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


def apply_response_text(model_response: ModelResponse, text: str, model: str) -> ModelResponse:
    """Populate litellm's pre-built ModelResponse in place with the ACP agent's reply."""
    choice = model_response.choices[0]
    choice.message.content = text
    choice.message.role = "assistant"
    choice.finish_reason = "stop"
    model_response.model = model
    return model_response
