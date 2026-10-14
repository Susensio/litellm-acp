# Backlog

- packaging | Publish to PyPI: needs a PyPI account and a trusted-publisher link to this GitHub repo (only the account owner can do this), plus a tag-triggered GitHub Actions release workflow.
- provider | Streamed calls log no cost: LiteLLM recomputes cost from the reassembled stream (`_response_cost_calculator` on the `stream_chunk_builder` result) and drops the adapter-reported cost the non-streaming path passes via `_hidden_params["response_cost"]`. Fixing it needs a supported hook in LiteLLM, or price-map entries per `acp/<name>`.
- provider | Each completion spawns a fresh adapter subprocess with no concurrency cap; a proxy under sustained load could exhaust processes/fds. No repro yet, just a scale concern worth a semaphore or a documented limit.
