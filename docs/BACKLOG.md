# Backlog

- packaging | Publish to PyPI: needs a PyPI account and a trusted-publisher link to this GitHub repo (only the account owner can do this), plus a tag-triggered GitHub Actions release workflow.
- provider | stream=True isn't implemented (astreaming/streaming raise) - already flagged in the README as a known limitation.
- provider | Each completion spawns a fresh adapter subprocess with no concurrency cap; a proxy under sustained load could exhaust processes/fds. No repro yet, just a scale concern worth a semaphore or a documented limit.
