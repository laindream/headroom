"""Pure policy for one deliberate cache-prefix rewrite near context exhaustion."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from headroom.cache.prefix_tracker import (
    CachePressureRejectionMemo,
    _canonicalize_for_prefix_compare,
)
from headroom.config import _MUTATING_TOOL_NAMES, _READ_TOOL_NAMES

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


@dataclass
class UpstreamContextRescue:
    """One forced full-history candidate authorized by an upstream overflow."""

    body: dict[str, Any]
    optimized_tokens: int
    transforms_applied: list[str]
    pipeline_timing: dict[str, float]
    waste_signals: dict[str, int] | None
    additional_latency_ms: float = 0.0


@dataclass(frozen=True)
class ClaudeContextEstimate:
    """Claude Code context estimate plus provenance for observability."""

    tokens: int
    source: str


@dataclass(frozen=True)
class CachePressureCandidateProjection:
    """Project a count-API delta into Claude Code's context-token space."""

    counted_tokens_saved: int
    projected_tokens: int
    projected_ratio: float


def cache_pressure_history_fingerprint(
    messages: list[dict[str, Any]],
) -> str:
    """Hash semantic history bytes without retaining prompt content in the memo."""

    canonical = _canonicalize_for_prefix_compare(messages)
    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _contains_candidate_eligibility_change(
    messages: list[dict[str, Any]],
    *,
    protect_recent_tool_result_turns: int,
) -> bool:
    lifecycle_tools = _READ_TOOL_NAMES | _MUTATING_TOOL_NAMES
    for message in messages:
        if not isinstance(message, dict):
            continue
        if protect_recent_tool_result_turns > 0 and message.get("role") == "tool":
            return True
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if (
                    protect_recent_tool_result_turns > 0
                    and isinstance(block, dict)
                    and block.get("type") == "tool_result"
                ):
                    # Any new tool-result turn can move a previously protected
                    # result beyond the recent-turn boundary, unlocking much
                    # more compression than the appended token count alone.
                    return True
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_use"
                    and block.get("name") in lifecycle_tools
                ):
                    return True
        for tool_call in message.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function") or {}
            if isinstance(function, dict) and function.get("name") in lifecycle_tools:
                return True
        function_call = message.get("function_call")
        if isinstance(function_call, dict) and function_call.get("name") in lifecycle_tools:
            return True
    return False


def should_retry_cache_pressure_candidate(
    memo: CachePressureRejectionMemo,
    messages: list[dict[str, Any]],
    *,
    pressure_tokens: int,
    max_output_ratio: float,
    protect_recent_tool_result_turns: int = 0,
) -> bool:
    """Retry only when a rejected candidate can plausibly cross the gate.

    New context is an upper bound on additional removable tokens. Read/edit
    events bypass that bound because they can supersede large historical
    observations without adding comparable text. A tool-result turn bypasses
    it only when positional recent-turn protection is enabled; the coding
    profile routes observations by content with a zero-turn window.
    """

    if pressure_tokens <= 0 or not 0 < max_output_ratio <= 1:
        return True
    if memo.max_output_ratio != max_output_ratio or len(messages) < memo.message_count:
        return True
    if (
        cache_pressure_history_fingerprint(messages[: memo.message_count])
        != memo.history_fingerprint
    ):
        return True

    appended = messages[memo.message_count :]
    if _contains_candidate_eligibility_change(
        appended,
        protect_recent_tool_result_turns=protect_recent_tool_result_turns,
    ):
        return True

    new_context_upper_bound = max(0, pressure_tokens - memo.pressure_tokens)
    possible_saved_tokens = memo.counted_tokens_saved + new_context_upper_bound
    required_saved_tokens = math.ceil((1 - max_output_ratio) * pressure_tokens)
    return possible_saved_tokens >= required_saved_tokens


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


def _canonicalize_for_usage_anchor(messages: list[dict[str, Any]]) -> Any:
    """Normalize assistant reasoning echo churn for token anchoring only.

    Claude Code may echo a prior thinking block as ``redacted_thinking`` or
    omit it entirely.  That representation change must remain significant for
    prefix replay, but it does not invalidate the provider-reported token total
    used as Claude Code's context anchor.  Visible text and tool calls remain
    exact in this looser, count-only comparison.
    """

    canonical = _canonicalize_for_prefix_compare(messages)
    if not isinstance(canonical, list):
        return canonical
    for message in canonical:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = [
                block
                for block in content
                if not isinstance(block, dict)
                or block.get("type") not in {"thinking", "redacted_thinking"}
            ]
    return canonical


def estimate_claude_context(
    messages: list[dict[str, Any]],
    *,
    model: str,
    previous_messages: list[dict[str, Any]] | None = None,
    latest_response_total_tokens: int | None = None,
) -> ClaudeContextEstimate:
    """Mirror Claude Code's response-usage anchor plus estimated tail."""
    chars_per_token = _claude_chars_per_token(model)
    can_anchor = (
        isinstance(latest_response_total_tokens, int)
        and not isinstance(latest_response_total_tokens, bool)
        and latest_response_total_tokens > 0
        and previous_messages
        and len(messages) >= len(previous_messages)
    )
    if can_anchor:
        current_prefix = messages[: len(previous_messages)]
        if _canonicalize_for_prefix_compare(current_prefix) == _canonicalize_for_prefix_compare(
            previous_messages
        ):
            source = "response_usage"
        elif _canonicalize_for_usage_anchor(current_prefix) == _canonicalize_for_usage_anchor(
            previous_messages
        ):
            source = "response_usage_relaxed"
        else:
            source = "full_estimate"
        if source != "full_estimate":
            tail = messages[len(previous_messages) :]
            return ClaudeContextEstimate(
                tokens=latest_response_total_tokens
                + sum(
                    _estimate_claude_content_tokens(message.get("content"), chars_per_token)
                    for message in tail
                    if isinstance(message, dict)
                ),
                source=source,
            )

    return ClaudeContextEstimate(
        tokens=sum(
            _estimate_claude_content_tokens(message.get("content"), chars_per_token)
            for message in messages
            if isinstance(message, dict)
        ),
        source="full_estimate",
    )


def estimate_claude_context_tokens(
    messages: list[dict[str, Any]],
    *,
    model: str,
    previous_messages: list[dict[str, Any]] | None = None,
    latest_response_total_tokens: int | None = None,
) -> int:
    """Backward-compatible integer wrapper around :func:`estimate_claude_context`."""

    return estimate_claude_context(
        messages,
        model=model,
        previous_messages=previous_messages,
        latest_response_total_tokens=latest_response_total_tokens,
    ).tokens


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
    *,
    pressure_tokens: int | None = None,
) -> bool:
    """Return whether a candidate saves enough tokens to justify a prefix rewrite."""
    if original_tokens <= 0 or candidate_tokens < 0 or not 0 < max_output_ratio <= 1:
        return False
    if pressure_tokens is None:
        return candidate_tokens <= original_tokens * max_output_ratio
    projection = project_cache_pressure_candidate(
        pressure_tokens=pressure_tokens,
        baseline_tokens=original_tokens,
        candidate_tokens=candidate_tokens,
    )
    return projection.counted_tokens_saved > 0 and projection.projected_ratio <= max_output_ratio


def project_cache_pressure_candidate(
    *,
    pressure_tokens: int,
    baseline_tokens: int,
    candidate_tokens: int,
) -> CachePressureCandidateProjection:
    """Apply exact-count savings to Claude's context estimate.

    The count endpoint may overcount invariant non-text payloads such as base64
    images.  Subtracting baseline and candidate counts cancels that shared
    error; applying only the delta to Claude's usage-anchored estimate keeps
    the acceptance ratio in the same token space as auto-compaction.
    """

    counted_tokens_saved = baseline_tokens - candidate_tokens
    projected_tokens = max(0, pressure_tokens - counted_tokens_saved)
    projected_ratio = projected_tokens / pressure_tokens if pressure_tokens > 0 else math.inf
    return CachePressureCandidateProjection(
        counted_tokens_saved=counted_tokens_saved,
        projected_tokens=projected_tokens,
        projected_ratio=projected_ratio,
    )


def is_upstream_context_overflow(status_code: int, error_body: Any) -> bool:
    """Recognize an explicit upstream context-window rejection.

    Token estimates are intentionally excluded: a rescue retry is authorized
    only by the provider's actual 400 response.
    """
    if status_code != 400:
        return False

    if isinstance(error_body, bytes):
        text = error_body.decode("utf-8", errors="replace")
    elif isinstance(error_body, str):
        text = error_body
    else:
        try:
            text = json.dumps(error_body, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(error_body)

    normalized = text.lower()
    return any(
        marker in normalized
        for marker in (
            "input exceeds the context window",
            "context window exceeded",
            "prompt is too long",
            "maximum context length",
            "exceed context limit",
            "context_length_exceeded",
        )
    )
