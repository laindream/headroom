"""Strict lossy-scope invariants for coding-agent traffic."""

from __future__ import annotations

from typing import Any

from headroom.proxy.server import HeadroomProxy, ProxyConfig
from headroom.transforms.content_router import ContentRouter, ContentRouterConfig


class _Tokenizer:
    def count_text(self, text: str) -> int:
        return max(1, len(str(text)) // 4)


def _long(label: str) -> str:
    return (f"{label} important exact content. " * 80).strip()


def _router(*, protect_recent_tool_result_turns: int = 0) -> tuple[ContentRouter, list[str]]:
    router = ContentRouter(
        ContentRouterConfig(
            force_kompress_all=True,
            min_chars_for_block_compression=10,
            lossy_tool_results_only=True,
            protect_recent_tool_result_turns=protect_recent_tool_result_turns,
            lossy_tool_allowlist=frozenset({"Bash"}),
        )
    )
    calls: list[str] = []

    def fake_compress_block_content(**kwargs: Any) -> tuple[str | None, bool]:
        calls.append(kwargs["strategy_label"])
        kwargs["transforms_applied"].append(f"router:{kwargs['strategy_label']}:fake")
        return "COMPRESSED", True

    router._compress_block_content = fake_compress_block_content  # type: ignore[method-assign]
    return router, calls


def _tool_turn(
    tool_id: str,
    tool_name: str,
    output: str,
    *,
    command: str = "pytest -q",
) -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": tool_name,
                    "input": {"command": command},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": output,
                }
            ],
        },
    ]


def test_strict_scope_compresses_only_tool_result_not_user_or_assistant_text() -> None:
    router, calls = _router()
    tool_output = _long("build log")
    user_decision = _long("USER AUTHORIZATION AND DECISION")
    assistant_plan = _long("PLAN TODO REVIEW CONCLUSION")
    messages = [
        *_tool_turn("tool-old", "Bash", tool_output),
        {"role": "user", "content": [{"type": "text", "text": user_decision}]},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_plan}],
        },
    ]

    result = router.apply(
        messages,
        _Tokenizer(),
        compress_user_messages=True,
        compress_assistant_text_blocks=True,
        force_kompress=True,
        target_ratio=0.10,
    )

    assert calls == ["tool_result"]
    assert result.messages[1]["content"][0]["content"] == "COMPRESSED"
    assert result.messages[2]["content"][0]["text"] == user_decision
    assert result.messages[3]["content"][0]["text"] == assistant_plan


def test_strict_scope_protects_recent_tool_result_turns() -> None:
    router, calls = _router(protect_recent_tool_result_turns=1)
    old_output = _long("old test log")
    recent_output = _long("recent test log")
    messages = [
        *_tool_turn("tool-old", "Bash", old_output),
        *_tool_turn("tool-recent", "Bash", recent_output),
    ]

    result = router.apply(messages, _Tokenizer(), force_kompress=True)

    assert calls == ["tool_result"]
    assert result.messages[1]["content"][0]["content"] == "COMPRESSED"
    assert result.messages[3]["content"][0]["content"] == recent_output


def test_strict_scope_default_denies_unknown_and_agent_tool_results() -> None:
    router, calls = _router()
    agent_review = _long("agent review conclusion and TODO")
    unknown_output = _long("unknown MCP state")
    bash_output = _long("old build log")
    messages = [
        *_tool_turn("tool-agent", "Agent", agent_review),
        *_tool_turn("tool-mcp", "mcp__custom__opaque", unknown_output),
        *_tool_turn("tool-bash", "Bash", bash_output),
    ]

    result = router.apply(messages, _Tokenizer(), force_kompress=True)

    assert calls == ["tool_result"]
    assert result.messages[1]["content"][0]["content"] == agent_review
    assert result.messages[3]["content"][0]["content"] == unknown_output
    assert result.messages[5]["content"][0]["content"] == "COMPRESSED"


def test_strict_scope_allows_only_disposable_shell_commands() -> None:
    router, calls = _router()
    plan_file = _long("authorization plan file")
    opaque_script = _long("unknown script output")
    test_log = _long("pytest log")
    messages = [
        *_tool_turn("tool-read", "Bash", plan_file, command="cat PLAN.md"),
        *_tool_turn(
            "tool-script",
            "Bash",
            opaque_script,
            command="python scripts/report.py",
        ),
        *_tool_turn("tool-test", "Bash", test_log, command="pytest -q"),
    ]

    result = router.apply(messages, _Tokenizer(), force_kompress=True)

    assert calls == ["tool_result"]
    assert result.messages[1]["content"][0]["content"] == plan_file
    assert result.messages[3]["content"][0]["content"] == opaque_script
    assert result.messages[5]["content"][0]["content"] == "COMPRESSED"


def test_strict_scope_protects_old_failed_tool_output_at_any_size() -> None:
    router, calls = _router()
    failure = ("ERROR: test failed with traceback and exception\n" * 400).strip()

    result = router.apply(
        _tool_turn("tool-failure", "Bash", failure, command="pytest -q"),
        _Tokenizer(),
        force_kompress=True,
    )

    assert calls == []
    assert result.messages[1]["content"][0]["content"] == failure


def test_strict_scope_fails_open_when_transform_mutates_protected_text() -> None:
    router, _calls = _router()
    original = _long("USER AUTHORIZATION")

    def corrupt_protected_text(
        message: dict[str, Any],
        content_blocks: list[Any],
        *_args: Any,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        return {**message, "content": [{"type": "text", "text": "CORRUPTED"}]}

    router._process_content_blocks = corrupt_protected_text  # type: ignore[method-assign]
    result = router.apply(
        [{"role": "user", "content": [{"type": "text", "text": original}]}],
        _Tokenizer(),
        compress_user_messages=True,
        force_kompress=True,
    )

    assert result.messages == [{"role": "user", "content": [{"type": "text", "text": original}]}]
    assert "router:strict_scope_rejected" in result.transforms_applied


def test_proxy_wires_strict_scope_and_disables_pre_router_read_replacement() -> None:
    proxy = HeadroomProxy(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            code_aware_enabled=False,
            lossy_tool_results_only=True,
            protect_recent_tool_result_turns=3,
            lossy_tool_allowlist=frozenset({"Bash", "bash"}),
        )
    )
    router = proxy.anthropic_pipeline.transforms[-1]

    assert router.config.lossy_tool_results_only is True
    assert router.config.protect_recent_tool_result_turns == 3
    assert router.config.lossy_tool_allowlist == frozenset({"Bash", "bash"})
    assert router.config.read_lifecycle.enabled is False


def test_coding_profile_enables_strict_scope_without_wrapper_env() -> None:
    proxy = HeadroomProxy(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            code_aware_enabled=False,
            savings_profile="coding",
        )
    )
    router = proxy.anthropic_pipeline.transforms[-1]

    assert router.config.lossy_tool_results_only is True
    assert router.config.lossy_tool_allowlist == frozenset({"Bash"})
    assert router.config.read_lifecycle.enabled is False
