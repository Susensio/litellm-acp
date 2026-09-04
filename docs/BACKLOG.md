# Backlog

- ci | No GitHub Actions workflow: ruff, ty and pytest aren't enforced on push/PR.
- ci | py.typed ships typed hints but nothing runs a type checker (ty) to verify them - can fold into the same CI workflow above.
- packaging | Publish to PyPI: needs a PyPI account and a trusted-publisher link to this GitHub repo (only the account owner can do this), plus a tag-triggered GitHub Actions release workflow.
- packaging | pyproject.toml's version and _version.py's fallback string are both hardcoded "0.1.0" in two places; consider hatch-vcs or python-semantic-release so they can't drift.
- provider | stream=True isn't implemented (astreaming/streaming raise) - already flagged in the README as a known limitation.
- provider | Unclear whether ModelResponse.usage gets populated from the ACP session; if not, LiteLLM cost tracking/budgets silently report zero cost for acp/* calls.
- provider | Each completion spawns a fresh adapter subprocess with no concurrency cap; a proxy under sustained load could exhaust processes/fds. No repro yet, just a scale concern worth a semaphore or a documented limit.
