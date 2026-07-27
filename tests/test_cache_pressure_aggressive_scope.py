from __future__ import annotations

from typing import Any

from headroom.transforms.content_detector import ContentType
from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
    RouterCompressionResult,
    RoutingDecision,
)


class _Tokenizer:
    @staticmethod
    def count_text(text: str) -> int:
        return max(1, len(str(text)) // 4)


def _text(label: str) -> str:
    return (label + " ") * 160


def _router() -> tuple[ContentRouter, list[str]]:
    router = ContentRouter(
        ContentRouterConfig(
            protect_tool_results=frozenset({"Write", "Agent"}),
            min_chars_for_block_compression=25,
        )
    )
    compressed: list[str] = []

    def fake_compress_block_content(**kwargs: Any) -> tuple[str | None, bool]:
        compressed.append(kwargs["content"])
        return "COMPRESSED <<ccr:test>>", True

    router._compress_block_content = fake_compress_block_content  # type: ignore[method-assign]
    return router, compressed


def _pressure_kwargs() -> dict[str, Any]:
    return {
        "force_kompress": True,
        "target_ratio": 0.10,
        "compress_user_messages": False,
        "compress_system_messages": False,
        "compress_assistant_text_blocks": True,
        "protect_recent_messages": 4,
        "protect_recent": 0,
        "protect_analysis_context": False,
        "protect_reads": False,
        "protect_error_outputs": False,
        "require_reversible_lossy": True,
        "exclude_tools": frozenset({"headroom_retrieve"}),
        "protect_tool_results": frozenset({"headroom_retrieve"}),
        "min_tokens_to_compress": 1,
        "min_chars_for_block_compression": 25,
    }


def test_pressure_scope_protects_recent_tail_but_compresses_old_assistant_and_tools() -> None:
    router, compressed = _router()
    messages = [
        {"role": "system", "content": _text("system")},
        {"role": "user", "content": _text("user intent")},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "old-write", "name": "Write", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "old-write", "content": _text("old write")}
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": _text("old reasoning")}]},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "recent", "name": "Read", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "recent", "content": _text("recent read")}
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": _text("recent reasoning")}]},
        {"role": "user", "content": _text("latest user")},
    ]

    result = router.apply(messages, _Tokenizer(), **_pressure_kwargs())

    assert result.messages[0] == messages[0]
    assert result.messages[1] == messages[1]
    assert result.messages[-4:] == messages[-4:]
    assert result.messages[3]["content"][0]["content"].startswith("COMPRESSED")
    assert result.messages[4]["content"][0]["text"].startswith("COMPRESSED")
    assert _text("old write") in compressed
    assert _text("old reasoning") in compressed


def test_pressure_scope_uses_compressible_token_budget_for_hot_tail() -> None:
    router, compressed = _router()
    payloads = [(f"message-{i} " * 40) for i in range(8)]
    messages = [
        {"role": "assistant", "content": [{"type": "text", "text": payload}]}
        for payload in payloads
    ]
    last_five_tokens = sum(_Tokenizer.count_text(payload) for payload in payloads[-5:])

    result = router.apply(
        messages,
        _Tokenizer(),
        **{
            **_pressure_kwargs(),
            "protect_recent_messages": 0,
            "protect_recent_tokens": last_five_tokens,
        },
    )

    assert result.messages[-5:] == messages[-5:]
    assert result.messages[2]["content"][0]["text"].startswith("COMPRESSED")
    assert payloads[2] in compressed


def test_pressure_scope_does_not_let_one_oversized_result_expand_hot_tail_budget() -> None:
    router, compressed = _router()
    oversized = "large tool result " * 200
    messages = [
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "read-1", "content": oversized}],
        }
    ]

    result = router.apply(
        messages,
        _Tokenizer(),
        **{
            **_pressure_kwargs(),
            "protect_recent_messages": 0,
            "protect_recent_tokens": 100,
        },
    )

    assert result.messages[0]["content"][0]["content"].startswith("COMPRESSED")
    assert oversized in compressed


def test_pressure_scope_uses_safer_target_for_old_assistant_than_tool_results() -> None:
    router = ContentRouter(ContentRouterConfig(min_chars_for_block_compression=25))
    observed: list[float | None] = []

    def compression(content: str, **_kwargs: Any) -> RouterCompressionResult:
        observed.append(router._runtime_target_ratio)
        return RouterCompressionResult(
            compressed="short <<ccr:test>>",
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
            routing_log=[],
        )

    router.compress = compression  # type: ignore[method-assign]
    messages = [
        {"role": "assistant", "content": [{"type": "text", "text": _text("reasoning")}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "read-1",
                    "content": _text("observation"),
                }
            ],
        },
    ]

    router.apply(
        messages,
        _Tokenizer(),
        **{
            **_pressure_kwargs(),
            "protect_recent_messages": 0,
            "assistant_target_ratio": 0.25,
        },
    )

    assert observed == [0.25, 0.10]


def test_pressure_assistant_target_survives_single_item_watchdog(
    monkeypatch,
) -> None:
    monkeypatch.setenv("HEADROOM_COMPRESSION_DEADLINE_MS", "1000")
    monkeypatch.setenv("HEADROOM_DETECT_BACKEND", "python")
    router = ContentRouter(ContentRouterConfig())
    observed: list[tuple[bool, float | None]] = []

    def compression(content: str, **_kwargs: Any) -> RouterCompressionResult:
        observed.append((router._runtime_force_kompress, router._runtime_target_ratio))
        return RouterCompressionResult(
            compressed="short <<ccr:test>>",
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
            routing_log=[
                RoutingDecision(
                    content_type=ContentType.PLAIN_TEXT,
                    strategy=CompressionStrategy.KOMPRESS,
                    original_tokens=100,
                    compressed_tokens=10,
                )
            ],
        )

    router.compress = compression  # type: ignore[method-assign]
    result = router.apply(
        [{"role": "assistant", "content": _text("cold reasoning")}],
        _Tokenizer(),
        **{
            **_pressure_kwargs(),
            "protect_recent_messages": 0,
            "assistant_target_ratio": 0.25,
        },
    )

    assert observed == [(True, 0.25)]
    assert result.messages[0]["content"].startswith("short")


def test_ccr_retrieve_result_is_hard_protected_under_pressure_with_plugin_name(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "HEADROOM_MCP_TOOL_PREFIX",
        "mcp__plugin_headroom_headroom__",
    )
    router, compressed = _router()
    original = _text("retrieved original evidence")
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "retrieve-1",
                    "name": "mcp__plugin_headroom_headroom__headroom_retrieve",
                    "input": {"hash": "abc"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "retrieve-1",
                    "content": original,
                }
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": _text("filler 1")}]},
        {"role": "assistant", "content": [{"type": "text", "text": _text("filler 2")}]},
        {"role": "assistant", "content": [{"type": "text", "text": _text("filler 3")}]},
        {"role": "assistant", "content": [{"type": "text", "text": _text("filler 4")}]},
    ]

    result = router.apply(messages, _Tokenizer(), **_pressure_kwargs())

    assert result.messages[1]["content"][0]["content"] == original
    assert original not in compressed
    assert "router:protected:ccr_retrieve_result" in result.transforms_applied


def test_pressure_scope_protects_user_even_if_router_default_allows_user_compression() -> None:
    router = ContentRouter(ContentRouterConfig(skip_user_messages=False))
    original = _text("authoritative user decision")

    result = router.apply(
        [{"role": "user", "content": original}],
        _Tokenizer(),
        **{**_pressure_kwargs(), "protect_recent_messages": 0},
    )

    assert result.messages[0]["content"] == original


def test_pressure_scope_rejects_unrecoverable_lossy_assistant_compression() -> None:
    router = ContentRouter(ContentRouterConfig(min_chars_for_block_compression=25))
    original = _text("cold assistant reasoning")

    def unmarked_compression(content: str, **_kwargs: Any) -> RouterCompressionResult:
        return RouterCompressionResult(
            compressed="short but unrecoverable",
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
            routing_log=[
                RoutingDecision(
                    content_type=ContentType.PLAIN_TEXT,
                    strategy=CompressionStrategy.KOMPRESS,
                    original_tokens=100,
                    compressed_tokens=10,
                )
            ],
        )

    router.compress = unmarked_compression  # type: ignore[method-assign]
    result = router.apply(
        [{"role": "assistant", "content": [{"type": "text", "text": original}]}],
        _Tokenizer(),
        **{**_pressure_kwargs(), "protect_recent_messages": 0},
    )

    assert result.messages[0]["content"][0]["text"] == original
    assert result.transforms_applied == ["router:noop"]
