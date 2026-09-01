"""litellm-acp: Bring-Your-Own-Subscription bridge for LiteLLM.

Importing this package registers the `acp` custom provider with LiteLLM, so:

    import litellm
    import litellm_acp

    response = litellm.completion(
        model="acp/claude",
        messages=[{"role": "user", "content": "hello"}],
    )

routes to the local `claude-agent-acp` CLI over stdio (ACP) instead of a
pay-per-token API. See `litellm_acp.register_adapter` to add more adapters.
"""

from __future__ import annotations

import litellm

from ._version import __version__
from .adapters import ADAPTER_REGISTRY, AdapterSpec, register_adapter
from .errors import ACPAdapterNotFoundError
from .provider import ACPCustomLLM

__all__ = [
    "ADAPTER_REGISTRY",
    "ACPAdapterNotFoundError",
    "ACPCustomLLM",
    "AdapterSpec",
    "__version__",
    "register_adapter",
]

acp_handler = ACPCustomLLM()

if not isinstance(getattr(litellm, "custom_provider_map", None), list):
    litellm.custom_provider_map = []

if not any(entry.get("provider") == "acp" for entry in litellm.custom_provider_map):
    litellm.custom_provider_map.append({"provider": "acp", "custom_handler": acp_handler})
