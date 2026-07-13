from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from headroom.config import is_tool_excluded
from headroom.mcp_tool_names import (
    DEFAULT_HEADROOM_MCP_TOOL_PREFIX,
    HEADROOM_MCP_TOOL_NAMES,
    accepted_headroom_mcp_tool_names,
    is_headroom_mcp_tool_name,
)
from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin
from headroom.proxy.handlers.streaming import StreamingMixin


@pytest.mark.parametrize("tool_name", sorted(HEADROOM_MCP_TOOL_NAMES))
def test_default_policy_accepts_bare_and_standalone_names(tool_name: str) -> None:
    assert DEFAULT_HEADROOM_MCP_TOOL_PREFIX == "mcp__headroom__"
    assert accepted_headroom_mcp_tool_names(tool_name) == (
        tool_name,
        f"mcp__headroom__{tool_name}",
    )
    assert is_headroom_mcp_tool_name(tool_name, tool_name)
    assert is_headroom_mcp_tool_name(f"mcp__headroom__{tool_name}", tool_name)


def test_custom_prefix_replaces_default_standalone_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_prefix = "mcp__plugin_headroom_headroom__"
    monkeypatch.setenv("HEADROOM_MCP_TOOL_PREFIX", custom_prefix)

    assert is_headroom_mcp_tool_name(
        f"{custom_prefix}headroom_retrieve",
        "headroom_retrieve",
    )
    assert is_headroom_mcp_tool_name("headroom_retrieve", "headroom_retrieve")
    assert not is_headroom_mcp_tool_name(
        "mcp__headroom__headroom_retrieve",
        "headroom_retrieve",
    )


def test_blank_prefix_falls_back_to_standalone_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADROOM_MCP_TOOL_PREFIX", "   ")

    assert is_headroom_mcp_tool_name(
        "mcp__headroom__headroom_retrieve",
        "headroom_retrieve",
    )


@pytest.mark.parametrize(
    "name",
    [
        "evil__headroom_retrieve",
        "mcp__headroom__headroom_retrieve_extra",
        "mcp__headroom__retrieve",
        "headroom_retrieve_extra",
        "",
        None,
    ],
)
def test_policy_rejects_unconfigured_or_inexact_names(name: object) -> None:
    assert not is_headroom_mcp_tool_name(name, "headroom_retrieve")


def test_anthropic_availability_accepts_default_standalone_tool() -> None:
    tools = [{"name": "mcp__headroom__headroom_retrieve", "input_schema": {}}]

    assert AnthropicHandlerMixin._has_headroom_retrieve_tool(tools)


def test_anthropic_availability_accepts_custom_tool_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADROOM_MCP_TOOL_PREFIX", "mcp__plugin_headroom_headroom__")
    tools = [
        {
            "name": "mcp__plugin_headroom_headroom__headroom_retrieve",
            "input_schema": {},
        }
    ]

    assert AnthropicHandlerMixin._has_headroom_retrieve_tool(tools)


def test_custom_prefix_participates_in_tool_exclusion_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADROOM_MCP_TOOL_PREFIX", "custom-headroom::")

    assert is_tool_excluded(
        "custom-headroom::headroom_retrieve",
        {"headroom_retrieve"},
    )


def test_anthropic_streaming_feedback_accepts_default_standalone_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MagicMock()
    monkeypatch.setattr(
        "headroom.cache.compression_store.get_compression_store",
        lambda: store,
    )
    response = {
        "content": [
            {
                "type": "tool_use",
                "name": "mcp__headroom__headroom_retrieve",
                "input": {"hash": "abc123def456"},
            }
        ]
    }

    StreamingMixin()._record_ccr_feedback_from_response(response, "anthropic", "req-1")

    store.retrieve.assert_called_once_with("abc123def456")


def test_openai_streaming_feedback_accepts_custom_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADROOM_MCP_TOOL_PREFIX", "custom-headroom::")
    store = MagicMock()
    monkeypatch.setattr(
        "headroom.cache.compression_store.get_compression_store",
        lambda: store,
    )
    payload = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {
                                "name": "custom-headroom::headroom_retrieve",
                                "arguments": '{"hash":"abc123def456"}',
                            },
                        }
                    ]
                }
            }
        ]
    }
    sse = f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n"

    StreamingMixin()._record_ccr_feedback_from_openai_sse(sse, "req-2")

    store.retrieve.assert_called_once_with("abc123def456")
