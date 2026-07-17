"""Claude-compatible pressure trigger; count-API candidates need a large win."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from headroom.cache.prefix_tracker import PrefixCacheTracker, _strip_cache_control
from headroom.cli.main import main
from headroom.proxy.cache_pressure_policy import (
    anthropic_usage_total_tokens,
    claude_auto_compact_threshold,
    estimate_claude_context_tokens,
    should_accept_cache_pressure_candidate,
    should_attempt_cache_pressure,
    should_rescue_cache_pressure_candidate,
)
from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin
from headroom.proxy.server import ProxyConfig, create_app
from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
    RouterCompressionResult,
)


def test_cache_pressure_threshold_matches_claude_effective_window() -> None:
    assert claude_auto_compact_threshold(372_000, 0.85) == 299_200
    assert not should_attempt_cache_pressure(299_199, 372_000, 0.85)
    assert should_attempt_cache_pressure(299_200, 372_000, 0.85)


def test_cache_pressure_default_trigger_ratio_is_85_percent() -> None:
    assert ProxyConfig().cache_pressure_trigger_ratio == 0.85


def test_claude_context_estimator_matches_custom_model_content_rules() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "abcdef"},
                {"type": "image", "source": {"type": "base64", "data": "ignored"}},
                {"type": "document", "source": {"type": "base64", "data": "ignored"}},
                {
                    "type": "tool_use",
                    "name": "lookup",
                    "input": {"emoji": "😀"},
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "tool-1",
                    "content": [{"type": "text", "text": "123456789"}],
                },
            ],
        }
    ]

    # gpt-5.6-sol is unknown to Claude Code, so text uses 3 UTF-16 code
    # units/token. Image/document blocks are fixed at 2,000 each.
    assert estimate_claude_context_tokens(messages, model="gpt-5.6-sol") == 4_012


def test_claude_context_estimator_uses_exact_known_model_set() -> None:
    message = [{"role": "user", "content": "x" * 8}]

    assert estimate_claude_context_tokens(message, model="claude-opus-4-6[1m]") == 2
    assert estimate_claude_context_tokens(message, model="claude-future-custom") == 3


def test_claude_context_estimator_uses_latest_api_usage_plus_appended_tail() -> None:
    previous = [
        {"role": "user", "content": "old question"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "old answer", "cache_control": {"type": "ephemeral"}}
            ],
        },
    ]
    current = [
        previous[0],
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "old answer"}],
        },
        {"role": "user", "content": "x" * 30},
    ]

    assert (
        estimate_claude_context_tokens(
            current,
            model="gpt-5.6-sol",
            previous_messages=previous,
            latest_response_total_tokens=290_000,
        )
        == 290_010
    )


def test_anthropic_usage_total_matches_claude_anchor_formula() -> None:
    assert (
        anthropic_usage_total_tokens(
            {
                "input_tokens": 887,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 329_216,
                "output_tokens": 84,
            }
        )
        == 330_187
    )
    assert anthropic_usage_total_tokens({"input_tokens": 887}) is None


def test_prefix_tracker_preserves_latest_response_total_tokens() -> None:
    tracker = PrefixCacheTracker("anthropic")
    tracker.update_from_response(
        cache_read_tokens=0,
        cache_write_tokens=0,
        messages=[{"role": "assistant", "content": "answer"}],
        response_total_tokens=330_187,
    )

    assert tracker.get_last_response_total_tokens() == 330_187


def test_streaming_message_delta_preserves_cli_proxy_usage_fields() -> None:
    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
        )
    )
    with TestClient(app) as client:
        usage = client.app.state.proxy._parse_sse_usage(
            b"event: message_delta\n"
            b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
            b'"usage":{"input_tokens":887,"cache_creation_input_tokens":0,'
            b'"cache_read_input_tokens":329216,"output_tokens":84}}\n\n',
            "anthropic",
        )

    assert usage == {
        "input_tokens": 887,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 329_216,
        "output_tokens": 84,
    }


def test_streaming_message_delta_does_not_erase_message_start_ttl_usage() -> None:
    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
        )
    )
    with TestClient(app) as client:
        usage = client.app.state.proxy._parse_sse_usage(
            b"event: message_start\n"
            b'data: {"type":"message_start","message":{"usage":{"input_tokens":887,'
            b'"cache_creation_input_tokens":1200,"cache_read_input_tokens":329216,'
            b'"cache_creation":{"ephemeral_5m_input_tokens":200,'
            b'"ephemeral_1h_input_tokens":1000}}}}\n\n'
            b"event: message_delta\n"
            b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
            b'"usage":{"output_tokens":84}}\n\n',
            "anthropic",
        )

    assert usage["cache_creation_ephemeral_5m_input_tokens"] == 200
    assert usage["cache_creation_ephemeral_1h_input_tokens"] == 1_000


def test_cache_pressure_candidate_requires_configured_reduction() -> None:
    assert should_accept_cache_pressure_candidate(350_000, 175_000, 0.50)
    assert not should_accept_cache_pressure_candidate(350_000, 175_001, 0.50)
    assert not should_accept_cache_pressure_candidate(0, 0, 0.50)
    assert not should_accept_cache_pressure_candidate(350_000, -1, 0.50)


def test_cache_pressure_candidate_rescues_only_a_hard_overflow() -> None:
    assert should_rescue_cache_pressure_candidate(404_166, 331_602, 352_000)
    assert not should_rescue_cache_pressure_candidate(352_000, 331_602, 352_000)
    assert not should_rescue_cache_pressure_candidate(404_166, 352_001, 352_000)
    assert not should_rescue_cache_pressure_candidate(404_166, -1, 352_000)


class _CountProxy(AnthropicHandlerMixin):
    ANTHROPIC_API_URL = "http://127.0.0.1:8317"

    def __init__(self, client: httpx.AsyncClient):
        self.http_client = client
        self.config = SimpleNamespace(cache_pressure_count_timeout_seconds=0.25)


@pytest.mark.asyncio
async def test_upstream_count_sends_complete_anthropic_request() -> None:
    body = {
        "model": "gpt-5.6-sol",
        "system": "system instructions",
        "tools": [
            {
                "name": "lookup",
                "description": "Look up a record",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 1024,
        "stream": True,
    }
    seen: dict[str, object] = {}

    def respond(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["api_key"] = request.headers.get("x-api-key")
        seen["anthropic_version"] = request.headers.get("anthropic-version")
        return httpx.Response(200, json={"input_tokens": 63})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        proxy = _CountProxy(client)
        tokens = await proxy._count_anthropic_request_tokens(
            body,
            {
                "x-api-key": "local-test-key",
                "anthropic-version": "2023-06-01",
            },
        )

    assert tokens == 63
    assert seen == {
        "url": "http://127.0.0.1:8317/v1/messages/count_tokens",
        "body": body,
        "api_key": "local-test-key",
        "anthropic_version": "2023-06-01",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(503, json={"error": "unavailable"}),
        httpx.Response(200, json={"input_tokens": "not-an-int"}),
        httpx.Response(200, json={}),
    ],
)
async def test_upstream_count_fails_closed_on_bad_response(response: httpx.Response) -> None:
    async def respond(_request: httpx.Request) -> httpx.Response:
        return response

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        proxy = _CountProxy(client)
        tokens = await proxy._count_anthropic_request_tokens(
            {"model": "gpt-5.6-sol", "messages": [{"role": "user", "content": "hi"}]},
            {"x-api-key": "local-test-key"},
        )

    assert tokens is None


@pytest.mark.asyncio
async def test_upstream_count_fails_closed_on_timeout() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("count timed out", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        proxy = _CountProxy(client)
        tokens = await proxy._count_anthropic_request_tokens(
            {"model": "gpt-5.6-sol", "messages": [{"role": "user", "content": "hi"}]},
            {"x-api-key": "local-test-key"},
        )

    assert tokens is None


def test_direct_server_env_reads_cache_pressure_policy(monkeypatch) -> None:
    from headroom.proxy.server import _proxy_config_from_env

    monkeypatch.setenv("HEADROOM_CACHE_PRESSURE_TOKEN_MODE", "1")
    monkeypatch.setenv("HEADROOM_CACHE_PRESSURE_TRIGGER_RATIO", "0.91")
    monkeypatch.setenv("HEADROOM_CACHE_PRESSURE_TARGET_RATIO", "0.12")
    monkeypatch.setenv("HEADROOM_CACHE_PRESSURE_MAX_OUTPUT_RATIO", "0.49")
    monkeypatch.setenv("HEADROOM_CACHE_PRESSURE_COUNT_TIMEOUT_SECONDS", "4.5")

    config = _proxy_config_from_env()

    assert config.cache_pressure_token_mode_enabled is True
    assert config.cache_pressure_trigger_ratio == 0.91
    assert config.cache_pressure_target_ratio == 0.12
    assert config.cache_pressure_max_output_ratio == 0.49
    assert config.cache_pressure_count_timeout_seconds == 4.5


def test_click_proxy_env_reads_cache_pressure_policy() -> None:
    captured: dict[str, object] = {}

    def run_server(config, **_kwargs) -> None:  # noqa: ANN001
        captured["config"] = config

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("headroom.proxy.server.run_server", run_server)
        result = CliRunner().invoke(
            main,
            ["proxy"],
            env={
                "HEADROOM_CACHE_PRESSURE_TOKEN_MODE": "1",
                "HEADROOM_CACHE_PRESSURE_TRIGGER_RATIO": "0.91",
                "HEADROOM_CACHE_PRESSURE_TARGET_RATIO": "0.12",
                "HEADROOM_CACHE_PRESSURE_MAX_OUTPUT_RATIO": "0.49",
                "HEADROOM_CACHE_PRESSURE_COUNT_TIMEOUT_SECONDS": "4.5",
            },
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert config.cache_pressure_token_mode_enabled is True
    assert config.cache_pressure_trigger_ratio == 0.91
    assert config.cache_pressure_target_ratio == 0.12
    assert config.cache_pressure_max_output_ratio == 0.49
    assert config.cache_pressure_count_timeout_seconds == 4.5


class _Tracker:
    def __init__(self) -> None:
        self._cached_token_count = 300_000
        self._idle_seconds_at_fetch = 0.0
        self._last_response_total_tokens = 300_000
        self.previous_original = [
            {"role": "user", "content": "original historical request"},
            {"role": "assistant", "content": "historical answer"},
        ]
        self.previous_forwarded = [
            {"role": "user", "content": "cached forwarded request"},
            {"role": "assistant", "content": "historical answer"},
        ]

    def get_frozen_message_count(self) -> int:
        return 2

    def get_last_original_messages(self):  # noqa: ANN201
        return self.previous_original.copy()

    def get_last_forwarded_messages(self):  # noqa: ANN201
        return self.previous_forwarded.copy()

    def get_last_response_total_tokens(self) -> int:
        return self._last_response_total_tokens

    def update_from_response(self, **kwargs) -> None:  # noqa: ANN003
        self.previous_original = kwargs["original_messages"].copy()
        self.previous_forwarded = kwargs["messages"].copy()
        self._last_response_total_tokens = kwargs.get("response_total_tokens")


def _result(messages, *, marker: str = "cache"):  # noqa: ANN001, ANN202
    return SimpleNamespace(
        messages=messages,
        transforms_applied=[marker],
        timing={},
        tokens_before=350_000,
        tokens_after=150_000,
        waste_signals=None,
    )


@pytest.mark.parametrize(
    (
        "pressure_tokens",
        "baseline_tokens",
        "candidate_tokens",
        "expected_first_content",
        "expected_decision",
    ),
    [
        (299_199, None, None, "cached forwarded request", "below_threshold"),
        (299_200, None, None, "cached forwarded request", "count_unavailable"),
        (299_200, 350_000, 227_500, "pressure-compressed history", "accepted"),
        (299_200, 350_000, 227_501, "cached forwarded request", "insufficient_reduction"),
        (299_200, 404_166, 331_602, "pressure-compressed history", "accepted"),
        (299_200, 350_000, None, "cached forwarded request", "candidate_count_unavailable"),
    ],
)
def test_cache_pressure_candidate_controls_prefix_overlay(
    pressure_tokens: int,
    baseline_tokens: int | None,
    candidate_tokens: int | None,
    expected_first_content: str,
    expected_decision: str,
) -> None:
    config = ProxyConfig(
        mode="cache",
        optimize=True,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        image_optimize=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        cache_pressure_token_mode_enabled=True,
        cache_pressure_trigger_ratio=0.85,
        cache_pressure_target_ratio=0.12,
        cache_pressure_max_output_ratio=0.65,
        cache_pressure_count_timeout_seconds=0.25,
    )
    app = create_app(config)
    captured: dict[str, object] = {}
    captured_logs: list[object] = []
    pressure_pipeline_kwargs: dict[str, object] = {}
    tracker = _Tracker()
    tracker._last_response_total_tokens = pressure_tokens - 4
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        proxy.session_tracker_store = SimpleNamespace(
            compute_session_id=lambda *_args, **_kwargs: "pressure-session",
            get_or_create=lambda *_args, **_kwargs: tracker,
        )
        proxy.anthropic_provider.get_context_limit = lambda _model: 372_000
        count_results = (
            [] if expected_decision == "below_threshold" else [baseline_tokens, candidate_tokens]
        )
        count_tokens = AsyncMock(side_effect=count_results)
        proxy._count_anthropic_request_tokens = count_tokens
        proxy.logger = SimpleNamespace(log=captured_logs.append)

        def apply_pipeline(**kwargs):  # noqa: ANN003, ANN202
            if kwargs["frozen_message_count"] == 0:
                pressure_pipeline_kwargs.update(kwargs)
                pressure_messages = [message.copy() for message in kwargs["messages"]]
                pressure_messages[0]["content"] = "pressure-compressed history"
                return _result(
                    pressure_messages,
                    marker="cache_pressure:token_mode",
                )
            return _result(kwargs["messages"])

        proxy.anthropic_pipeline.apply = apply_pipeline

        async def fake_retry(method, url, headers, body, **kwargs):  # noqa: ANN001, ANN202
            captured["body"] = body
            return httpx.Response(
                200,
                json={
                    "id": "msg_pressure",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 10,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = fake_retry

        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "gpt-5.6-sol",
                "max_tokens": 128,
                "messages": tracker.previous_original
                + [{"role": "user", "content": "new live turn"}],
            },
        )

    assert response.status_code == 200
    sent_messages = captured["body"]["messages"]
    assert sent_messages[0]["content"] == expected_first_content
    assert tracker.previous_forwarded[0]["content"] == expected_first_content
    expected_count_calls = (
        0 if expected_decision == "below_threshold" else 1 if baseline_tokens is None else 2
    )
    assert count_tokens.await_count == expected_count_calls
    if expected_decision in {"below_threshold", "count_unavailable"}:
        assert pressure_pipeline_kwargs == {}
    else:
        assert pressure_pipeline_kwargs["target_ratio"] == 0.12
        assert pressure_pipeline_kwargs["force_kompress"] is True
    assert len(captured_logs) == 1
    tags = captured_logs[0].tags
    assert tags["cache_pressure_decision"] == expected_decision
    assert tags["cache_pressure_trigger_tokens"] == pressure_tokens
    assert tags["cache_pressure_effective_context_limit"] == 352_000
    assert tags["cache_pressure_trigger_threshold_tokens"] == 299_200
    assert tags["cache_pressure_context_usage_ratio"] == round(pressure_tokens / 352_000, 6)
    if baseline_tokens is None:
        assert "cache_pressure_baseline_tokens" not in tags
    else:
        assert tags["cache_pressure_baseline_tokens"] == baseline_tokens
    if candidate_tokens is None:
        assert "cache_pressure_candidate_tokens" not in tags
        assert "cache_pressure_candidate_ratio" not in tags
    else:
        assert tags["cache_pressure_candidate_tokens"] == candidate_tokens
        assert tags["cache_pressure_candidate_ratio"] == round(
            candidate_tokens / baseline_tokens,
            6,
        )
    if expected_decision in {"below_threshold", "count_unavailable"}:
        assert "cache_pressure_target_ratio" not in tags
    else:
        assert tags["cache_pressure_target_ratio"] == 0.12


def test_accepted_pressure_prefix_is_reused_by_next_cache_turn(monkeypatch) -> None:
    """A one-turn prefix rewrite becomes the next cache turn's immutable prefix."""

    monkeypatch.setenv("HEADROOM_MCP_TOOL_PREFIX", "mcp__plugin_headroom_headroom__")
    config = ProxyConfig(
        mode="cache",
        optimize=True,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        image_optimize=False,
        ccr_inject_tool=True,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        cache_pressure_token_mode_enabled=True,
        cache_pressure_trigger_ratio=0.85,
        cache_pressure_target_ratio=0.10,
        cache_pressure_max_output_ratio=0.65,
        cache_pressure_count_timeout_seconds=0.25,
    )
    app = create_app(config)
    tracker = _Tracker()
    retrieve_tool = {
        "name": "mcp__plugin_headroom_headroom__headroom_retrieve",
        "description": "Retrieve compressed context",
        "input_schema": {
            "type": "object",
            "properties": {"hash": {"type": "string"}},
            "required": ["hash"],
        },
    }
    sent_bodies: list[dict[str, object]] = []
    pipeline_inputs: list[tuple[int, list[dict[str, object]]]] = []

    with TestClient(app) as client:
        proxy = client.app.state.proxy
        proxy.session_tracker_store = SimpleNamespace(
            compute_session_id=lambda *_args, **_kwargs: "pressure-transition-session",
            get_or_create=lambda *_args, **_kwargs: tracker,
        )
        proxy.anthropic_provider.get_context_limit = lambda _model: 372_000
        proxy._count_anthropic_request_tokens = AsyncMock(side_effect=[350_000, 150_000])

        def apply_pipeline(**kwargs):  # noqa: ANN003, ANN202
            messages = [message.copy() for message in kwargs["messages"]]
            frozen = kwargs["frozen_message_count"]
            pipeline_inputs.append((frozen, messages))
            if frozen == 0:
                # Deliberately collapse the whole request to one message. The
                # production router is currently 1:1, but the transition must
                # remain correct if a future token transform merges messages.
                return _result(
                    [
                        {
                            "role": "user",
                            "content": "pressure-compressed full request "
                            "[Retrieve more: hash=abcdef123456abcdef123456]",
                        }
                    ],
                    marker="cache_pressure:token_mode",
                )
            return _result(messages)

        proxy.anthropic_pipeline.apply = apply_pipeline

        async def fake_retry(method, url, headers, body, **kwargs):  # noqa: ANN001, ANN202
            sent_bodies.append(json.loads(json.dumps(body)))
            return httpx.Response(
                200,
                json={
                    "id": f"msg_pressure_{len(sent_bodies)}",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"answer {len(sent_bodies)}"}],
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 10,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = fake_retry

        first = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "gpt-5.6-sol",
                "max_tokens": 128,
                "tools": [retrieve_tool],
                "messages": tracker.previous_original
                + [{"role": "user", "content": "pressure-triggering live turn"}],
            },
        )
        next_original = tracker.previous_original + [
            {"role": "user", "content": "first post-pressure cache delta"}
        ]
        second = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "gpt-5.6-sol",
                "max_tokens": 128,
                "tools": [retrieve_tool],
                "messages": next_original,
            },
        )
        third_original = tracker.previous_original + [
            {"role": "user", "content": "second post-pressure cache delta"}
        ]
        third = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "gpt-5.6-sol",
                "max_tokens": 128,
                "tools": [retrieve_tool],
                "messages": third_original,
            },
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 200
    assert len(sent_bodies) == 3
    first_forwarded = sent_bodies[0]["messages"]
    second_forwarded = sent_bodies[1]["messages"]
    third_forwarded = sent_bodies[2]["messages"]
    assert len(first_forwarded) == 1
    assert second_forwarded[0] == first_forwarded[0]
    assert second_forwarded[1]["role"] == "assistant"
    assert second_forwarded[1]["content"][0]["text"] == "answer 1"
    assert second_forwarded[-1] == {
        "role": "user",
        "content": "first post-pressure cache delta",
    }
    assert _strip_cache_control(third_forwarded[: len(second_forwarded)]) == (
        _strip_cache_control(second_forwarded)
    )
    assert third_forwarded[len(second_forwarded)]["role"] == "assistant"
    assert third_forwarded[len(second_forwarded)]["content"][0]["text"] == "answer 2"
    assert third_forwarded[-1] == {
        "role": "user",
        "content": "second post-pressure cache delta",
    }
    assert sent_bodies[0]["tools"] == [retrieve_tool]
    assert sent_bodies[1]["tools"] == [retrieve_tool]
    assert sent_bodies[2]["tools"] == [retrieve_tool]
    for forwarded in (second_forwarded, third_forwarded):
        assert (
            sum(
                1
                for message in forwarded
                for block in (
                    message.get("content") if isinstance(message.get("content"), list) else []
                )
                if "cache_control" in block
            )
            == 1
        )
    assert len(tracker.previous_original) != len(tracker.previous_forwarded)
    assert tracker.previous_forwarded[: len(third_forwarded)] == third_forwarded
    assert proxy._count_anthropic_request_tokens.await_count == 2
    assert [frozen for frozen, _messages in pipeline_inputs] == [2, 0, 2, 4]


def test_content_router_request_overrides_are_isolated_across_concurrent_calls(
    monkeypatch,
) -> None:
    """A pressure request must not leak its aggressive runtime profile to cache traffic."""

    monkeypatch.setenv("HEADROOM_COMPRESS_WORKERS", "2")

    class WordTokenizer:
        @staticmethod
        def count_text(value: str) -> int:
            return len(value.split())

    router = ContentRouter(ContentRouterConfig(enable_kompress=False))
    start_barrier = threading.Barrier(2)
    compress_barrier = threading.Barrier(4)
    observed: dict[str, list[tuple[float | None, bool]]] = {
        "pressure": [],
        "cache": [],
    }

    def fake_compress(content, **_kwargs):  # noqa: ANN001, ANN202
        request_kind = "pressure" if content.startswith("pressure") else "cache"
        compress_barrier.wait(timeout=3)
        observed[request_kind].append(
            (
                router._runtime_target_ratio,
                router._runtime_force_kompress,
            )
        )
        return RouterCompressionResult(
            compressed=content,
            original=content,
            strategy_used=CompressionStrategy.PASSTHROUGH,
        )

    router.compress = fake_compress

    def apply(kind: str, *, target_ratio: float | None, force_kompress: bool) -> None:
        start_barrier.wait(timeout=3)
        router.apply(
            messages=[
                {"role": "user", "content": f"{kind}-{index} " + "word " * 80} for index in range(2)
            ],
            tokenizer=WordTokenizer(),
            model_limit=372_000,
            compress_user_messages=True,
            target_ratio=target_ratio,
            force_kompress=force_kompress,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        pressure = executor.submit(
            apply,
            "pressure",
            target_ratio=0.10,
            force_kompress=True,
        )
        cache = executor.submit(
            apply,
            "cache",
            target_ratio=None,
            force_kompress=False,
        )
        pressure.result(timeout=3)
        cache.result(timeout=3)

    assert observed == {
        "pressure": [(0.10, True), (0.10, True)],
        "cache": [(None, False), (None, False)],
    }
