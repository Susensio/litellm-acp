"""Optional end-to-end smoke test for a locally authenticated ACP adapter."""

import os

import litellm
import pytest

import litellm_acp  # noqa: F401 - import side effect registers the "acp" provider


@pytest.mark.real_adapter
def test_real_adapter_returns_text():
    adapter = os.environ.get("LITELLM_ACP_SMOKE_MODEL")
    if adapter is None:
        pytest.skip("set LITELLM_ACP_SMOKE_MODEL to run against a real ACP adapter")

    response = litellm.completion(
        model=f"acp/{adapter}",
        messages=[{"role": "user", "content": "Reply with exactly: litellm-acp smoke test"}],
        timeout=60,
    )

    assert response.choices[0].message.content.strip()
