import sys
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fake_agent_argv() -> list[str]:
    """[binary, *args] to spawn the bundled fake ACP agent as an AdapterSpec would."""
    return [sys.executable, str(FIXTURES_DIR / "fake_acp_agent.py")]
