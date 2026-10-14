# ADR-0001: Run each ACP turn behind one LiteLLM-free module as an event iterator, not a handle object or provider-side code

Status: Accepted
Date: 2026-10-14

## Context

By October 2026 one ACP turn (spawn an adapter, handshake, prompt, collect, tear down) was spread over five functions in `provider.py` and a collector class in `client.py`.
Its results came back three ways: a returned `(text, usage, cost)` tuple, an `on_text` callback, and a `cost` attribute set on the collector.
Adding `stream=True` had built a second path around it with its own queue.
Lifecycle behaviour (shared deadline, teardown on early close) could only be tested through LiteLLM, which forced `gc.collect()` tricks.
The next backlog item, a cap on concurrent adapter processes, had no single place to go.

Alternatives weighed:
- **A handle object** with `.text_deltas()` and an awaitable `.result()`: two ways in, and teardown is hard to guarantee when only one is consumed.
- **An `async with` interface**: LiteLLM iterates whatever `astreaming` returns and never calls `aclose()`, so a context manager could not survive the hand-off.
- **Raising LiteLLM's `CustomLLMError` inside the turn**: less mapping code, but LiteLLM would leak into the module and its tests.

## Decision

One module, `litellm_acp.turn`, owns the whole turn behind `run_turn(adapter, prompt, timeout)`, a plain async generator yielding `TextDelta`s then one `TurnEnd(usage, cost, stop_reason)`.
It raises its own `TurnTimeoutError` / `TurnFailedError` and never imports LiteLLM.
It takes ACP prompt blocks, not OpenAI messages.
The headless ACP `Client` is private to it.

The provider is the only code that knows LiteLLM: it flattens messages, maps turn errors to `CustomLLMError` (504 / 500), and shapes events into a `ModelResponse` or streaming chunks.
Completion drains the iterator; streaming yields from it.

Landed in commit `e221425`; `client.py` was deleted there.

## Consequences

- Subprocess lifecycle concerns (spawning, deadlines, stderr, teardown, and a future concurrency cap) belong in the turn module, not the provider.
- Lifecycle tests run against `run_turn` directly, without LiteLLM.
- Anything the agent reports per turn surfaces through `TurnEnd`; `stop_reason` became `finish_reason` that way in the following commit.
- Teardown relies on generator closing. Consumers inside litellm-acp wrap the iterator in `contextlib.aclosing`; an external consumer that abandons it without closing leaves the adapter running until the generator is garbage-collected.
- Only the two turn errors get mapped: errors the agent itself reports propagate as the ACP SDK raised them, as before the change.

## Corrections

- 2026-10-14: the Decision said the provider "shapes events into a `ModelResponse` or streaming chunks". Later the same day that shaping, with the $0 cost policy, moved into `translate.py` alongside message flattening, leaving `provider.py` with the entry points, adapter resolution and error mapping. The turn module staying LiteLLM-free was unaffected.
