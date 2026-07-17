from __future__ import annotations

from headroom.ccr.tool_calls import (
    CCRToolCall,
    extract_tool_calls,
    has_ccr_tool_calls,
    parse_ccr_tool_calls,
    should_handle_ccr_server_side,
    tool_call_id_for_provider,
)
from headroom.ccr.tool_injection import CCR_TOOL_NAME

HASH = "abc123def456abc123def456"


def test_extract_tool_calls_handles_provider_shapes() -> None:
    anthropic = {"content": [{"type": "tool_use", "id": "t1", "name": CCR_TOOL_NAME}]}
    openai = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"id": "c1", "function": {"name": CCR_TOOL_NAME, "arguments": "{}"}}
                    ]
                }
            }
        ]
    }
    google = {
        "candidates": [
            {"content": {"parts": [{"functionCall": {"name": CCR_TOOL_NAME, "args": {}}}]}}
        ]
    }
    responses = {"output": [{"type": "function_call", "name": CCR_TOOL_NAME}]}

    assert len(extract_tool_calls(anthropic, "anthropic")) == 1
    assert len(extract_tool_calls(openai, "openai")) == 1
    assert len(extract_tool_calls(google, "google")) == 1
    assert len(extract_tool_calls(responses, "openai_responses")) == 1


def test_extract_tool_calls_rejects_invalid_shapes() -> None:
    assert extract_tool_calls({"content": "not-a-list"}, "anthropic") == []
    assert extract_tool_calls({"choices": []}, "openai") == []
    assert extract_tool_calls({"choices": ["bad"]}, "openai") == []
    assert extract_tool_calls({"candidates": [{"content": {"parts": "bad"}}]}, "google") == []
    assert extract_tool_calls({"output": "bad"}, "openai_responses") == []
    assert extract_tool_calls({}, "unknown") == []


def test_has_ccr_tool_calls_uses_provider_native_names() -> None:
    assert has_ccr_tool_calls(
        {"content": [{"type": "tool_use", "name": CCR_TOOL_NAME, "input": {"hash": HASH}}]},
        "anthropic",
    )
    assert not has_ccr_tool_calls(
        {"content": [{"type": "tool_use", "name": "read_file", "input": {"hash": HASH}}]},
        "anthropic",
    )


def test_server_side_ccr_handling_requires_proxy_owned_tool() -> None:
    assert should_handle_ccr_server_side(
        response_handler_enabled=True,
        client_had_ccr_tool=False,
        forwarded_has_ccr_tool=True,
    )
    assert not should_handle_ccr_server_side(
        response_handler_enabled=True,
        client_had_ccr_tool=True,
        forwarded_has_ccr_tool=True,
    )
    assert not should_handle_ccr_server_side(
        response_handler_enabled=True,
        client_had_ccr_tool=False,
        forwarded_has_ccr_tool=False,
    )
    assert not should_handle_ccr_server_side(
        response_handler_enabled=False,
        client_had_ccr_tool=False,
        forwarded_has_ccr_tool=True,
    )


def test_has_ccr_tool_calls_accepts_default_standalone_mcp_name() -> None:
    response = {
        "content": [
            {
                "type": "tool_use",
                "id": "tool_1",
                "name": f"mcp__headroom__{CCR_TOOL_NAME}",
                "input": {"hash": HASH},
            }
        ]
    }

    assert has_ccr_tool_calls(response, "anthropic")
    ccr_calls, other_calls = parse_ccr_tool_calls(response, "anthropic")
    assert ccr_calls == [CCRToolCall(tool_call_id="tool_1", hash_key=HASH)]
    assert other_calls == []


def test_has_ccr_tool_calls_accepts_custom_mcp_prefix(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("HEADROOM_MCP_TOOL_PREFIX", "custom-headroom::")
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": f"custom-headroom::{CCR_TOOL_NAME}",
                                "arguments": '{"hash":"abc123def456abc123def456"}',
                            },
                        }
                    ]
                }
            }
        ]
    }

    assert has_ccr_tool_calls(response, "openai")


def test_parse_ccr_tool_calls_splits_retrievals_from_other_tools() -> None:
    response = {
        "content": [
            {"type": "tool_use", "id": "tool_1", "name": CCR_TOOL_NAME, "input": {"hash": HASH}},
            {"type": "tool_use", "id": "tool_2", "name": "read_file", "input": {"path": "a.py"}},
        ]
    }

    ccr_calls, other_calls = parse_ccr_tool_calls(response, "anthropic")

    assert ccr_calls == [CCRToolCall(tool_call_id="tool_1", hash_key=HASH)]
    assert other_calls == [
        {"type": "tool_use", "id": "tool_2", "name": "read_file", "input": {"path": "a.py"}}
    ]


def test_tool_call_id_for_provider_models_matching_result_ids() -> None:
    assert (
        tool_call_id_for_provider({"functionCall": {"name": CCR_TOOL_NAME}}, "google")
        == CCR_TOOL_NAME
    )
    assert (
        tool_call_id_for_provider({"id": "item_1", "call_id": "call_1"}, "openai_responses")
        == "call_1"
    )
    assert tool_call_id_for_provider({"id": "tool_1"}, "anthropic") == "tool_1"
