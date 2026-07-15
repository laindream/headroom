"""Cache-pressure escalation uses upstream counts and breaks cache only for a large win."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from headroom.cli.main import main
from headroom.proxy.cache_pressure_policy import (
    should_accept_cache_pressure_candidate,
    should_attempt_cache_pressure,
)
from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin
from headroom.proxy.server import ProxyConfig, create_app


def test_cache_pressure_threshold_uses_model_context_limit() -> None:
    assert not should_attempt_cache_pressure(334_799, 372_000, 0.90)
    assert should_attempt_cache_pressure(334_800, 372_000, 0.90)


def test_cache_pressure_candidate_requires_configured_reduction() -> None:
    assert should_accept_cache_pressure_candidate(350_000, 175_000, 0.50)
    assert not should_accept_cache_pressure_candidate(350_000, 175_001, 0.50)
    assert not should_accept_cache_pressure_candidate(0, 0, 0.50)
    assert not should_accept_cache_pressure_candidate(350_000, -1, 0.50)


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
    monkeypatch.setenv("HEADROOM_CACHE_PRESSURE_MAX_OUTPUT_RATIO", "0.49")
    monkeypatch.setenv("HEADROOM_CACHE_PRESSURE_COUNT_TIMEOUT_SECONDS", "4.5")

    config = _proxy_config_from_env()

    assert config.cache_pressure_token_mode_enabled is True
    assert config.cache_pressure_trigger_ratio == 0.91
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
                "HEADROOM_CACHE_PRESSURE_MAX_OUTPUT_RATIO": "0.49",
                "HEADROOM_CACHE_PRESSURE_COUNT_TIMEOUT_SECONDS": "4.5",
            },
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert config.cache_pressure_token_mode_enabled is True
    assert config.cache_pressure_trigger_ratio == 0.91
    assert config.cache_pressure_max_output_ratio == 0.49
    assert config.cache_pressure_count_timeout_seconds == 4.5


class _Tracker:
    def __init__(self) -> None:
        self._cached_token_count = 300_000
        self._idle_seconds_at_fetch = 0.0
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

    def update_from_response(self, **kwargs) -> None:  # noqa: ANN003
        self.previous_original = kwargs["original_messages"].copy()
        self.previous_forwarded = kwargs["messages"].copy()


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
        "baseline_tokens",
        "candidate_tokens",
        "expected_first_content",
        "expected_decision",
    ),
    [
        (334_799, None, "cached forwarded request", "below_threshold"),
        (350_000, 175_000, "pressure-compressed history", "accepted"),
        (350_000, 175_001, "cached forwarded request", "insufficient_reduction"),
        (350_000, None, "cached forwarded request", "candidate_count_unavailable"),
    ],
)
def test_cache_pressure_candidate_controls_prefix_overlay(
    baseline_tokens: int,
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
        cache_pressure_trigger_ratio=0.90,
        cache_pressure_max_output_ratio=0.50,
        cache_pressure_count_timeout_seconds=0.25,
    )
    app = create_app(config)
    captured: dict[str, object] = {}
    captured_logs: list[object] = []
    tracker = _Tracker()

    with TestClient(app) as client:
        proxy = client.app.state.proxy
        proxy.session_tracker_store = SimpleNamespace(
            compute_session_id=lambda *_args, **_kwargs: "pressure-session",
            get_or_create=lambda *_args, **_kwargs: tracker,
        )
        proxy.anthropic_provider.get_context_limit = lambda _model: 372_000
        count_results = [baseline_tokens]
        if expected_decision != "below_threshold":
            count_results.append(candidate_tokens)
        count_tokens = AsyncMock(side_effect=count_results)
        proxy._count_anthropic_request_tokens = count_tokens
        proxy.logger = SimpleNamespace(log=captured_logs.append)

        def apply_pipeline(**kwargs):  # noqa: ANN003, ANN202
            if kwargs["frozen_message_count"] == 0:
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
    assert count_tokens.await_count == (1 if expected_decision == "below_threshold" else 2)
    assert len(captured_logs) == 1
    tags = captured_logs[0].tags
    assert tags["cache_pressure_decision"] == expected_decision
    assert tags["cache_pressure_baseline_tokens"] == baseline_tokens
    assert tags["cache_pressure_context_usage_ratio"] == round(baseline_tokens / 372_000, 6)
    if candidate_tokens is None:
        assert "cache_pressure_candidate_tokens" not in tags
        assert "cache_pressure_candidate_ratio" not in tags
    else:
        assert tags["cache_pressure_candidate_tokens"] == candidate_tokens
        assert tags["cache_pressure_candidate_ratio"] == round(
            candidate_tokens / baseline_tokens,
            6,
        )
