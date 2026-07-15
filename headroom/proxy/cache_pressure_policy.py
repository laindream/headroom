"""Pure policy for one deliberate cache-prefix rewrite near context exhaustion."""

from __future__ import annotations


def should_attempt_cache_pressure(
    original_tokens: int,
    context_limit: int,
    trigger_ratio: float,
) -> bool:
    """Return whether a cache-mode request is close enough to the context limit."""
    if original_tokens <= 0 or context_limit <= 0 or not 0 < trigger_ratio <= 1:
        return False
    return original_tokens >= context_limit * trigger_ratio


def should_accept_cache_pressure_candidate(
    original_tokens: int,
    candidate_tokens: int,
    max_output_ratio: float,
) -> bool:
    """Return whether a candidate saves enough tokens to justify a prefix rewrite."""
    if original_tokens <= 0 or candidate_tokens < 0 or not 0 < max_output_ratio <= 1:
        return False
    return candidate_tokens <= original_tokens * max_output_ratio
