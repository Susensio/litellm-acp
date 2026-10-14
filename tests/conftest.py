import asyncio
import os
import sys
from pathlib import Path

import litellm
import pytest
from litellm.integrations.custom_logger import CustomLogger

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fake_agent_argv() -> list[str]:
    """[binary, *args] to spawn the bundled fake ACP agent as an AdapterSpec would."""
    return [sys.executable, str(FIXTURES_DIR / "fake_acp_agent.py")]


class _CostRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.costs: dict[str, list[float | None]] = {}

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        logged = kwargs["standard_logging_object"]
        self.costs.setdefault(logged["model"], []).append(logged["response_cost"])

    async def wait_for(self, model: str, timeout: float = 5) -> list[float | None]:
        """Success callbacks run in the background after the call returns."""
        async with asyncio.timeout(timeout):
            while model not in self.costs:
                await asyncio.sleep(0.05)
        return self.costs[model]


@pytest.fixture(scope="session")
def logged_costs():
    """The `response_cost` LiteLLM logs per model, as budgets see it.

    Registered once: LiteLLM keeps callbacks it has already set up, so swapping
    `litellm.callbacks` per test doesn't take. Use a model name unique to the test.
    """
    recorder = _CostRecorder()
    litellm.callbacks.append(recorder)
    return recorder
