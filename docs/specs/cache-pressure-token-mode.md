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
Protect the active working set by a bounded token budget, keep more cold
assistant reasoning than replaceable tool observations, and skip lossy model
work on blocks too small to amortize its latency and CCR marker overhead.

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
- `headroom/transforms/content_router.py`: hot-tail and value-tier routing
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
- Router tests prove that the protected tail is bounded by compressible tokens,
  authority/protocol bytes do not consume that budget, assistant and tool
  targets stay request-local under parallel compression, and an oversized tool
  result remains CCR-compressible instead of expanding the hot tail.
- Streaming finalizer tests prove compression overhead is included in total
  request latency.
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
- Always: preserve the configured request-count cooldown after an accepted
  rewrite. Once it expires, the normal Claude-compatible threshold remains the
  re-arm gate, so elapsed turns alone never authorize another rewrite.

## Success criteria

- Default cache behavior remains byte-for-byte compatible when feature is off.
- Below the configured trigger ratio, no token-mode candidate is generated.
- Below the configured trigger ratio, no count-API request is made.
- At or above it, the candidate receives the configured pressure target ratio,
  forces Kompress, and is accepted only when
  the configured projected whole-request ratio (Armory uses `0.80`).
- Count timeout/error/malformed response keeps cache-mode output.
- `below_threshold` logs Claude-compatible trigger tokens and effective-window
  utilization. Accepted and insufficient-reduction decisions also log the
  count-API baseline, candidate tokens, and candidate/baseline ratio.
- Accepted candidate is forwarded without old-prefix overlay; next request can
  freeze the newly forwarded prefix through existing tracker logic.
- With token-tail protection enabled, only the newest contiguous set of lossy-
  eligible assistant/tool observations that fits the configured budget stays
  exact. User/system text and tool protocol remain exact independently of that
  budget; one oversized observation is compressed with CCR rather than silently
  expanding the budget.
- The pressure profile can use a safer assistant keep ratio than the common
  tool-result ratio and can set a minimum lossy-content size; byte/data-lossless
  folds remain eligible below that floor.
- Streaming `total_latency_ms` includes `optimization_latency_ms`, maintaining
  the invariant `total_latency_ms >= overhead_ms`.
- With Armory's `270000` policy window, trigger `0.85` fires at `212500` tokens:
  `floor((270000 - 20000) * 0.85)`, before Claude's configured `0.95` line.
- Armory pins the tested fork commit and enables trigger `0.85`, tool-result
  target `0.10`, assistant target `0.25`, maximum whole-request output `0.80`,
  a `16000`-token hot tail, and a `128`-token lossy floor for the local
  `gpt-5.6-sol` profile.

## Open questions

None. Ratios and workload-dependent hot-tail/floor values remain configurable.
