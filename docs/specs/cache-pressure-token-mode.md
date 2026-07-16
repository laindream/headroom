# Spec: Cache-pressure token-mode escalation

## Objective

Keep Anthropic proxy traffic in cache mode during normal operation. When a
request reaches the same context-pressure line Claude Code uses, allow one
deliberate prefix-cache rewrite only when a full-history token-mode candidate
removes enough input. Mirror Claude Code's latest-response usage anchor plus
locally estimated appended tail for the trigger. Only after that trigger, use
the configured Anthropic token-count endpoint for both the forwarded baseline
and candidate. Generate the candidate with an explicit per-content target ratio
and forced Kompress while retaining the active profile's safety protections.

## Tech stack

- Python 3.10+
- FastAPI proxy with `httpx.AsyncClient`
- CLIProxyAPI-compatible `POST /v1/messages/count_tokens`
- pytest / pytest-asyncio

## Commands

```bash
pytest -q tests/test_cache_pressure_token_mode.py
ruff check headroom tests
ruff format --check headroom tests
```

## Project structure

- `headroom/proxy/cache_pressure_policy.py`: pure threshold/acceptance policy
- `headroom/proxy/handlers/anthropic.py`: upstream count call and request flow
- `headroom/proxy/models.py`: opt-in configuration
- `tests/test_cache_pressure_token_mode.py`: policy and count-boundary tests

## Code style

```python
effective_window = context_limit - 20_000
trigger_tokens = min(
    floor(effective_window * trigger_ratio),
    effective_window - 13_000,
)
if claude_context_tokens < trigger_tokens:
    return False
return candidate_tokens <= int(original_tokens * max_output_ratio)
```

Prefer pure policy functions. Network failures remain request-local and never
authorize a cache-breaking rewrite.

## Testing strategy

- Unit-test Claude-compatible usage anchoring, local tail estimation, the
  effective-window threshold boundary, and candidate acceptance.
- Async-test full request body, auth headers, response parsing, timeout, and
  malformed/error responses with `httpx.MockTransport` or a stub client.
- Handler regression test proves an accepted escalation bypasses cached-prefix
  overlay exactly once; rejected/failed counts preserve cache mode.
- Handler regression test proves requests below the pressure line do not call
  the count endpoint. Triggered requests record count-API baseline and
  candidate/baseline reduction without extra work.
- Run focused tests first, then existing Anthropic/cache-mode suites.

## Boundaries

- When triggered: count the complete Anthropic request
  (`system`, `tools`, `messages`).
- Always: use Claude Code's response usage formula
  (`input + cache_creation + cache_read + output`) for the trigger anchor.
- Always: default feature off in upstream-compatible code; Armory opts in.
- Always: keep configured cache mode after the pressure request completes.
- Always: treat the pressure target ratio as candidate-generation guidance;
  authorize rewrites only from complete-request counts returned by the
  configured count API.
- Ask first: changing CLIProxyAPI or its auth/config.
- Never: send credentials to a different host than configured Anthropic
  upstream; log auth header values; use Headroom's heuristic counter to
  authorize a prefix rewrite.

## Success criteria

- Default cache behavior remains byte-for-byte compatible when feature is off.
- Below the configured trigger ratio, no token-mode candidate is generated.
- Below the configured trigger ratio, no count-API request is made.
- At or above it, the candidate receives the configured pressure target ratio,
  forces Kompress, and is accepted only when
  `candidate/original <= 0.65`.
- Count timeout/error/malformed response keeps cache-mode output.
- `below_threshold` logs Claude-compatible trigger tokens and effective-window
  utilization. Accepted and insufficient-reduction decisions also log the
  count-API baseline, candidate tokens, and candidate/baseline ratio.
- Accepted candidate is forwarded without old-prefix overlay; next request can
  freeze the newly forwarded prefix through existing tracker logic.
- With a `372000` context window, trigger `0.85` fires at `299200` tokens:
  `floor((372000 - 20000) * 0.85)`, before Claude's configured `0.95` line.
- Armory pins the tested fork commit and enables trigger `0.85`, pressure target
  `0.10`, and maximum whole-request output `0.65` for the local `gpt-5.6-sol`
  profile.

## Open questions

None. Ratios remain configurable; Armory defaults are trigger `0.85`, pressure
target `0.10`, maximum whole-request output `0.65`.
