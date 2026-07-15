# Spec: Cache-pressure token-mode escalation

## Objective

Keep Anthropic proxy traffic in cache mode during normal operation. When a
request approaches the model context limit, allow one deliberate prefix-cache
rewrite only when a full-history token-mode candidate removes enough input.
Generate that candidate with an explicit per-content target ratio and forced
Kompress while retaining the active profile's safety protections.
Use the configured Anthropic upstream token-count endpoint instead of
Headroom's heuristic counter for both sides of that decision.

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
if original_tokens < int(context_limit * trigger_ratio):
    return False
return candidate_tokens <= int(original_tokens * max_output_ratio)
```

Prefer pure policy functions. Network failures remain request-local and never
authorize a cache-breaking rewrite.

## Testing strategy

- Unit-test threshold boundary and 50% acceptance using pure functions.
- Async-test full request body, auth headers, response parsing, timeout, and
  malformed/error responses with `httpx.MockTransport` or a stub client.
- Handler regression test proves an accepted escalation bypasses cached-prefix
  overlay exactly once; rejected/failed counts preserve cache mode.
- Handler regression test proves every successful baseline count records exact
  baseline/context utilization, and every successful candidate count records
  exact candidate/baseline reduction without extra work.
- Run focused tests first, then existing Anthropic/cache-mode suites.

## Boundaries

- Always: count the complete Anthropic request (`system`, `tools`, `messages`).
- Always: default feature off in upstream-compatible code; Armory opts in.
- Always: keep configured cache mode after the pressure request completes.
- Always: treat the pressure target ratio as candidate-generation guidance;
  authorize rewrites only from exact whole-request counts.
- Ask first: changing CLIProxyAPI or its auth/config.
- Never: send credentials to a different host than configured Anthropic
  upstream; log auth header values; use heuristic counts to authorize a prefix
  rewrite.

## Success criteria

- Default cache behavior remains byte-for-byte compatible when feature is off.
- Below the configured trigger ratio, no token-mode candidate is generated.
- At or above it, the candidate receives the configured pressure target ratio,
  forces Kompress, and is accepted only when
  `candidate/original <= 0.50`.
- Count timeout/error/malformed response keeps cache-mode output.
- `below_threshold` logs exact baseline tokens and context utilization without
  generating a candidate. Accepted and insufficient-reduction decisions also
  log exact candidate tokens and candidate/baseline ratio when that count
  succeeds.
- Accepted candidate is forwarded without old-prefix overlay; next request can
  freeze the newly forwarded prefix through existing tracker logic.
- Armory pins the tested fork commit and enables trigger `0.80`, pressure target
  `0.10`, and maximum whole-request output `0.50` for the local `gpt-5.6-sol`
  profile.

## Open questions

None. Ratios remain configurable; Armory defaults are trigger `0.80`, pressure
target `0.10`, maximum whole-request output `0.50`.
