"""Pure policy for one deliberate cache-prefix rewrite near context exhaustion."""

from __future__ import annotations

import json
import math
from typing import Any

from headroom.cache.prefix_tracker import _canonicalize_for_prefix_compare

CLAUDE_OUTPUT_RESERVE_TOKENS = 20_000
CLAUDE_COMPACT_HARD_RESERVE_TOKENS = 13_000
CLAUDE_FOUR_CHARS_PER_TOKEN_MODELS = frozenset(
    {
        "claude-3-opus",
        "claude-3-sonnet",
        "claude-3-haiku",
        "claude-3-5-sonnet",
        "claude-3-5-haiku",
        "claude-3-7-sonnet",
        "claude-opus-4-0",
        "claude-opus-4-1",
        "claude-opus-4-5",
        "claude-opus-4-6",
        "claude-sonnet-4-0",
        "claude-sonnet-4-5",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    }
)


def _claude_chars_per_token(model: str) -> int:
    """Match Claude Code's local estimator for resolved model ids."""
    normalized = model.strip().lower()
    if normalized.endswith("[1m]"):
        normalized = normalized[:-4]
    normalized = normalized.replace("_", "-").replace(".", "-")
    return 4 if normalized in CLAUDE_FOUR_CHARS_PER_TOKEN_MODELS else 3


def _javascript_string_length(value: str) -> int:
    """Return JavaScript ``String.length`` (UTF-16 code units)."""
    return len(value.encode("utf-16-le", errors="surrogatepass")) // 2


def _claude_estimated_text_tokens(value: Any, chars_per_token: int) -> int:
    if not isinstance(value, str) or chars_per_token <= 0:
        return 0
    units = _javascript_string_length(value)
    # JavaScript Math.round for a non-negative quotient.
    return (2 * units + chars_per_token) // (2 * chars_per_token)


def _javascript_json_stringify(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _estimate_claude_content_tokens(content: Any, chars_per_token: int) -> int:
    if not content:
        return 0
    if isinstance(content, str):
        return _claude_estimated_text_tokens(content, chars_per_token)
    if not isinstance(content, list):
        return _claude_estimated_text_tokens(
            _javascript_json_stringify(content),
            chars_per_token,
        )

    total = 0
    for block in content:
        if isinstance(block, str):
            total += _claude_estimated_text_tokens(block, chars_per_token)
            continue
        if not isinstance(block, dict):
            total += _claude_estimated_text_tokens(
                _javascript_json_stringify(block),
                chars_per_token,
            )
            continue

        block_type = block.get("type")
        if block_type == "text":
            total += _claude_estimated_text_tokens(block.get("text"), chars_per_token)
        elif block_type in {"image", "document"}:
            total += 2_000
        elif block_type == "tool_result":
            total += _estimate_claude_content_tokens(block.get("content"), chars_per_token)
        elif block_type == "tool_use":
            serialized = str(block.get("name", "")) + _javascript_json_stringify(
                block.get("input") or {}
            )
            total += _claude_estimated_text_tokens(serialized, chars_per_token)
        elif block_type == "thinking":
            total += _claude_estimated_text_tokens(block.get("thinking"), chars_per_token)
        elif block_type == "redacted_thinking":
            total += _claude_estimated_text_tokens(block.get("data"), chars_per_token)
        else:
            total += _claude_estimated_text_tokens(
                _javascript_json_stringify(block),
                chars_per_token,
            )
    return total


def estimate_claude_context_tokens(
    messages: list[dict[str, Any]],
    *,
    model: str,
    previous_messages: list[dict[str, Any]] | None = None,
    latest_response_total_tokens: int | None = None,
) -> int:
    """Mirror Claude Code's response-usage anchor plus estimated tail."""
    chars_per_token = _claude_chars_per_token(model)
    tail = messages

    if (
        isinstance(latest_response_total_tokens, int)
        and not isinstance(latest_response_total_tokens, bool)
        and latest_response_total_tokens > 0
        and previous_messages
        and len(messages) >= len(previous_messages)
        and _canonicalize_for_prefix_compare(messages[: len(previous_messages)])
        == _canonicalize_for_prefix_compare(previous_messages)
    ):
        tail = messages[len(previous_messages) :]
        return latest_response_total_tokens + sum(
            _estimate_claude_content_tokens(message.get("content"), chars_per_token)
            for message in tail
            if isinstance(message, dict)
        )

    return sum(
        _estimate_claude_content_tokens(message.get("content"), chars_per_token)
        for message in tail
        if isinstance(message, dict)
    )


def anthropic_usage_total_tokens(usage: Any) -> int | None:
    """Return the same usage total Claude Code trusts as its context anchor."""
    if not isinstance(usage, dict):
        return None

    values: list[int] = []
    for key, required in (
        ("input_tokens", True),
        ("cache_creation_input_tokens", False),
        ("cache_read_input_tokens", False),
        ("output_tokens", True),
    ):
        value = usage.get(key, 0)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or (required and key not in usage)
        ):
            return None
        values.append(value)

    total = sum(values)
    return total if total > 0 else None


def claude_auto_compact_threshold(
    context_limit: int,
    trigger_ratio: float,
) -> int:
    """Apply Claude Code's output reserve before its percentage threshold."""
    if context_limit <= 0 or not 0 < trigger_ratio <= 1:
        return 0
    effective_window = claude_effective_context_limit(context_limit)
    percent_threshold = math.floor(effective_window * trigger_ratio)
    hard_threshold = max(0, effective_window - CLAUDE_COMPACT_HARD_RESERVE_TOKENS)
    return min(percent_threshold, hard_threshold)


def claude_effective_context_limit(context_limit: int) -> int:
    return max(0, context_limit - CLAUDE_OUTPUT_RESERVE_TOKENS)


def should_attempt_cache_pressure(
    original_tokens: int,
    context_limit: int,
    trigger_ratio: float,
) -> bool:
    """Return whether Claude Code's context estimate crossed the pressure line."""
    if original_tokens <= 0 or context_limit <= 0 or not 0 < trigger_ratio <= 1:
        return False
    return original_tokens >= claude_auto_compact_threshold(context_limit, trigger_ratio)


def should_accept_cache_pressure_candidate(
    original_tokens: int,
    candidate_tokens: int,
    max_output_ratio: float,
) -> bool:
    """Return whether a candidate saves enough tokens to justify a prefix rewrite."""
    if original_tokens <= 0 or candidate_tokens < 0 or not 0 < max_output_ratio <= 1:
        return False
    return candidate_tokens <= original_tokens * max_output_ratio
