"""Regression tests for Anthropic streaming CCR retrieval interception."""

from __future__ import annotations

import asyncio
import json
import logging
from unittest.mock import AsyncMock, patch

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.responses import StreamingResponse  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from starlette.requests import Request  # noqa: E402

from headroom.cache.compression_store import get_compression_store  # noqa: E402
from headroom.ccr.tool_injection import create_ccr_tool_definition  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


def _make_config() -> ProxyConfig:
    return ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=True,
        ccr_handle_responses=True,
        ccr_context_tracking=False,
        image_optimize=False,
    )


def _message_response(content: list[dict], *, stop_reason: str = "end_turn") -> dict:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": content,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    }


def _plugin_ccr_tool() -> dict:
    tool = create_ccr_tool_definition("anthropic")
    tool["name"] = "mcp__plugin_headroom_headroom__headroom_retrieve"
    return tool


def _inject_proxy_ccr_tool(**kwargs) -> tuple[list[dict], bool]:  # noqa: ANN003
    return list(kwargs.get("existing_tools") or []) + [
        create_ccr_tool_definition("anthropic")
    ], True


def _is_client_visible_sse(body: bytes) -> bool:
    return b"event:" in body or b"data:" in body


class _ContinuationClient:
    def __init__(self, response_json: dict) -> None:
        self.response_json = response_json
        self.post_calls: list[dict] = []

    async def post(self, url, *, content=None, headers=None, timeout=None):  # noqa: ANN001
        self.post_calls.append(
            {
                "url": url,
                "content": content,
                "headers": dict(headers or {}),
                "timeout": timeout,
            }
        )
        return httpx.Response(200, json=self.response_json)

    async def aclose(self) -> None:
        return None


def test_streaming_proxy_injected_headroom_retrieve_is_intercepted_and_returned_as_sse() -> None:
    config = _make_config()
    store = get_compression_store()
    hash_key = store.store(
        original=json.dumps({"secret": "retrieved answer"}),
        compressed="{}",
        original_item_count=1,
    )
    initial_response = _message_response(
        [
            {
                "type": "tool_use",
                "id": "toolu_ccr",
                "name": "headroom_retrieve",
                "input": {"hash": hash_key},
            }
        ],
        stop_reason="tool_use",
    )
    final_response = _message_response(
        [{"type": "text", "text": "retrieved answer is now available"}]
    )

    with (
        patch("headroom.proxy.server.AnyLLMBackend"),
        patch(
            "headroom.proxy.helpers.apply_session_sticky_ccr_tool",
            side_effect=_inject_proxy_ccr_tool,
        ),
    ):
        app = create_app(config)
        with TestClient(app) as client:
            proxy = client.app.state.proxy
            proxy._stream_response = AsyncMock(
                side_effect=AssertionError("live streaming path should not be used")
            )
            continuation_client = _ContinuationClient(final_response)
            proxy.http_client = continuation_client
            initial_bodies: list[dict] = []

            async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
                initial_bodies.append(json.loads(json.dumps(body)))
                assert stream is False
                assert body["stream"] is False
                return httpx.Response(200, json=initial_response)

            proxy._retry_request = _fake_retry  # type: ignore[assignment]

            resp = client.post(
                "/v1/messages",
                headers={
                    "x-api-key": "test-key",
                    "anthropic-version": "2023-06-01",
                    "accept": "text/event-stream",
                    "content-encoding": "identity",
                },
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 64,
                    "stream": True,
                    "messages": [{"role": "user", "content": "retrieve it"}],
                },
            )

    assert resp.status_code == 200, resp.text
    assert "text/event-stream" in resp.headers["content-type"]
    assert "retrieved answer is now available" in resp.text
    assert "headroom_retrieve" not in resp.text
    assert initial_bodies and initial_bodies[0]["stream"] is False
    assert len(continuation_client.post_calls) == 1
    continuation_body = json.loads(continuation_client.post_calls[0]["content"].decode())
    assert continuation_body["stream"] is False
    continuation_headers = {
        key.lower(): value for key, value in continuation_client.post_calls[0]["headers"].items()
    }
    assert "content-length" not in continuation_headers
    assert "content-encoding" not in continuation_headers
    assert "transfer-encoding" not in continuation_headers
    assert "accept-encoding" not in continuation_headers


def test_streaming_without_headroom_retrieve_uses_normal_streaming_path() -> None:
    config = _make_config()

    with patch("headroom.proxy.server.AnyLLMBackend"):
        app = create_app(config)
        with TestClient(app) as client:
            proxy = client.app.state.proxy

            async def _fake_stream_response(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
                async def _gen():
                    yield b"event: message_stop\n"
                    yield b'data: {"type":"message_stop"}\n\n'

                return StreamingResponse(_gen(), media_type="text/event-stream")

            proxy._stream_response = AsyncMock(side_effect=_fake_stream_response)

            resp = client.post(
                "/v1/messages",
                headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 64,
                    "stream": True,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

    assert resp.status_code == 200, resp.text
    assert "text/event-stream" in resp.headers["content-type"]
    assert '"message_stop"' in resp.text
    proxy._stream_response.assert_awaited_once()


def test_streaming_with_client_owned_headroom_retrieve_uses_live_stream(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_MCP_TOOL_PREFIX", "mcp__plugin_headroom_headroom__")
    config = _make_config()

    with patch("headroom.proxy.server.AnyLLMBackend"):
        app = create_app(config)
        with TestClient(app) as client:
            proxy = client.app.state.proxy

            async def _fake_stream_response(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
                assert args[2]["stream"] is True

                async def _gen():
                    yield b"event: message_stop\n"
                    yield b'data: {"type":"message_stop"}\n\n'

                return StreamingResponse(_gen(), media_type="text/event-stream")

            proxy._stream_response = AsyncMock(side_effect=_fake_stream_response)
            proxy._retry_request = AsyncMock(
                side_effect=AssertionError("client-owned CCR must not use buffered upstream")
            )

            resp = client.post(
                "/v1/messages",
                headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 64,
                    "stream": True,
                    "tools": [_plugin_ccr_tool()],
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

    assert resp.status_code == 200, resp.text
    assert "text/event-stream" in resp.headers["content-type"]
    assert '"message_stop"' in resp.text
    proxy._stream_response.assert_awaited_once()
    proxy._retry_request.assert_not_awaited()


def test_nonstreaming_client_owned_headroom_retrieve_is_returned_to_client(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_MCP_TOOL_PREFIX", "mcp__plugin_headroom_headroom__")
    config = _make_config()
    store = get_compression_store()
    hash_key = store.store(
        original=json.dumps({"secret": "client should retrieve this"}),
        compressed="{}",
        original_item_count=1,
    )
    client_tool_name = "mcp__plugin_headroom_headroom__headroom_retrieve"
    initial_response = _message_response(
        [
            {
                "type": "tool_use",
                "id": "toolu_client_ccr",
                "name": client_tool_name,
                "input": {"hash": hash_key},
            }
        ],
        stop_reason="tool_use",
    )

    with patch("headroom.proxy.server.AnyLLMBackend"):
        app = create_app(config)
        with TestClient(app) as client:
            proxy = client.app.state.proxy
            continuation_client = _ContinuationClient(
                _message_response([{"type": "text", "text": "proxy continuation"}])
            )
            proxy.http_client = continuation_client
            proxy._retry_request = AsyncMock(
                return_value=httpx.Response(200, json=initial_response)
            )

            resp = client.post(
                "/v1/messages",
                headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 64,
                    "stream": False,
                    "tools": [_plugin_ccr_tool()],
                    "messages": [{"role": "user", "content": "retrieve it"}],
                },
            )

    assert resp.status_code == 200, resp.text
    assert resp.json()["content"][0]["name"] == client_tool_name
    assert continuation_client.post_calls == []


def test_mixed_ccr_and_client_tool_streams_both_blocks_as_sse() -> None:
    """LEGAL mixed turn (#839, #2089): headroom_retrieve emitted alongside a
    client tool. The proxy cannot synthesize the client tool_result, so it must
    hand the turn back for the client to resolve — a 200 SSE stream preserving
    BOTH tool_use blocks, matching the non-streaming path. It must NOT 502 and
    must NOT issue a continuation request."""
    config = _make_config()
    initial_response = _message_response(
        [
            {
                "type": "tool_use",
                "id": "toolu_ccr",
                "name": "headroom_retrieve",
                "input": {"hash": "abc123abc123abc123abc123"},
            },
            {
                "type": "tool_use",
                "id": "toolu_client",
                "name": "client_tool",
                "input": {"value": 1},
            },
        ],
        stop_reason="tool_use",
    )

    with (
        patch("headroom.proxy.server.AnyLLMBackend"),
        patch(
            "headroom.proxy.helpers.apply_session_sticky_ccr_tool",
            side_effect=_inject_proxy_ccr_tool,
        ),
    ):
        app = create_app(config)
        with TestClient(app) as client:
            proxy = client.app.state.proxy
            proxy._stream_response = AsyncMock(
                side_effect=AssertionError("live streaming path should not be used")
            )
            continuation_client = _ContinuationClient(_message_response([]))
            proxy.http_client = continuation_client

            async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
                assert body["stream"] is False
                return httpx.Response(200, json=initial_response)

            proxy._retry_request = _fake_retry  # type: ignore[assignment]

            resp = client.post(
                "/v1/messages",
                headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 64,
                    "stream": True,
                    "tools": [
                        {
                            "name": "client_tool",
                            "description": "Client-owned tool",
                            "input_schema": {"type": "object", "properties": {}},
                        },
                    ],
                    "messages": [{"role": "user", "content": "use tools"}],
                },
            )

    assert resp.status_code == 200, resp.text
    assert "text/event-stream" in resp.headers["content-type"]
    # Both tool_use blocks are preserved for the client to resolve.
    assert "headroom_retrieve" in resp.text
    assert "client_tool" in resp.text
    assert "toolu_ccr" in resp.text
    assert "toolu_client" in resp.text
    assert "Unable to safely complete streamed CCR retrieval" not in resp.text
    # No continuation is issued — the client resolves all tool calls.
    assert continuation_client.post_calls == []


def test_unresolved_proxy_owned_ccr_fails_closed_as_502() -> None:
    """A proxy-injected CCR tool must not leak unresolved to an unaware client.

    Client-owned CCR tools stay on the live streaming path (covered above).
    This request does not declare the tool, so Headroom owns execution; if all
    continuation rounds still re-emit it, the only safe result is a 502 SSE.
    """
    config = _make_config()
    persistent_ccr = _message_response(
        [
            {
                "type": "tool_use",
                "id": "toolu_ccr",
                "name": "headroom_retrieve",
                "input": {"hash": "deadbeefdeadbeefdeadbeef"},
            },
        ],
        stop_reason="tool_use",
    )

    with (
        patch("headroom.proxy.server.AnyLLMBackend"),
        patch(
            "headroom.proxy.helpers.apply_session_sticky_ccr_tool",
            side_effect=_inject_proxy_ccr_tool,
        ),
    ):
        app = create_app(config)
        with TestClient(app) as client:
            proxy = client.app.state.proxy
            proxy._stream_response = AsyncMock(
                side_effect=AssertionError("live streaming path should not be used")
            )
            # Every continuation re-emits headroom_retrieve, so it never resolves.
            continuation_client = _ContinuationClient(persistent_ccr)
            proxy.http_client = continuation_client

            async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
                return httpx.Response(200, json=persistent_ccr)

            proxy._retry_request = _fake_retry  # type: ignore[assignment]

            resp = client.post(
                "/v1/messages",
                headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 64,
                    "stream": True,
                    "messages": [{"role": "user", "content": "use tools"}],
                },
            )

    assert resp.status_code == 502, resp.text
    assert "text/event-stream" in resp.headers["content-type"]
    assert "Unable to safely complete streamed CCR retrieval" in resp.text


@pytest.mark.asyncio
async def test_buffered_ccr_emits_keepalive_before_delayed_upstream() -> None:
    config = _make_config()
    final_response = _message_response([{"type": "text", "text": "done"}])
    started = asyncio.Event()
    release = asyncio.Event()
    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "stream": True,
        "messages": [{"role": "user", "content": "wait"}],
    }

    async def receive():
        return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/messages",
        "raw_path": b"/v1/messages",
        "query_string": b"",
        "headers": [(b"x-api-key", b"test-key"), (b"anthropic-version", b"2023-06-01")],
        "server": ("testserver", 80),
        "client": ("testclient", 123),
        "root_path": "",
    }

    with (
        patch("headroom.proxy.server.AnyLLMBackend"),
        patch(
            "headroom.proxy.helpers.apply_session_sticky_ccr_tool",
            side_effect=_inject_proxy_ccr_tool,
        ),
    ):
        app = create_app(config)
        with TestClient(app):
            proxy = app.state.proxy

            async def delayed_retry(*args, **kwargs):  # noqa: ANN002, ANN003
                started.set()
                await release.wait()
                return httpx.Response(200, json=final_response)

            proxy._retry_request = delayed_retry
            task = asyncio.create_task(proxy.handle_anthropic_messages(Request(scope, receive)))
            await started.wait()
            response = await asyncio.wait_for(asyncio.shield(task), 1)
            events: list[dict] = []
            first_visible_body = asyncio.Event()

            async def send(message):  # noqa: ANN001
                events.append(message)
                if message["type"] == "http.response.body" and _is_client_visible_sse(
                    message["body"]
                ):
                    first_visible_body.set()

            response_task = asyncio.create_task(response(scope, receive, send))
            await asyncio.wait_for(first_visible_body.wait(), 2)
            assert not release.is_set()
            release.set()
            await response_task

    bodies = [event["body"] for event in events if event["type"] == "http.response.body"]
    assert bodies[0] == b'event: ping\ndata: {"type":"ping"}\n\n'
    assert b"done" in b"".join(bodies)


@pytest.mark.asyncio
async def test_buffered_ccr_preserves_early_failure_status_and_headers() -> None:
    config = _make_config()
    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "stream": True,
        "messages": [{"role": "user", "content": "fail early"}],
    }

    async def receive():
        return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/messages",
        "raw_path": b"/v1/messages",
        "query_string": b"",
        "headers": [(b"x-api-key", b"test-key"), (b"anthropic-version", b"2023-06-01")],
        "server": ("testserver", 80),
        "client": ("testclient", 123),
        "root_path": "",
    }

    with (
        patch("headroom.proxy.server.AnyLLMBackend"),
        patch(
            "headroom.proxy.helpers.apply_session_sticky_ccr_tool",
            side_effect=_inject_proxy_ccr_tool,
        ),
    ):
        app = create_app(config)
        with TestClient(app):
            proxy = app.state.proxy

            async def early_failure(*args, **kwargs):  # noqa: ANN002, ANN003
                await asyncio.sleep(0.05)
                return httpx.Response(
                    429,
                    headers={"retry-after": "7"},
                    json={"error": {"message": "slow down"}},
                )

            proxy._retry_request = early_failure
            response = await proxy.handle_anthropic_messages(Request(scope, receive))
            events: list[dict] = []

            async def send(message):  # noqa: ANN001
                events.append(message)

            await response(scope, receive, send)

    start = next(event for event in events if event["type"] == "http.response.start")
    headers = dict(start["headers"])
    assert start["status"] == 429
    assert headers[b"retry-after"] == b"7"
    assert b": headroom-keepalive\n\n" not in b"".join(
        event["body"] for event in events if event["type"] == "http.response.body"
    )


@pytest.mark.asyncio
async def test_buffered_ccr_late_failure_emits_sanitized_error_event() -> None:
    config = _make_config()
    started = asyncio.Event()
    release = asyncio.Event()
    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "stream": True,
        "messages": [{"role": "user", "content": "wait"}],
    }

    async def receive():
        return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/messages",
        "raw_path": b"/v1/messages",
        "query_string": b"",
        "headers": [(b"x-api-key", b"test-key"), (b"anthropic-version", b"2023-06-01")],
        "server": ("testserver", 80),
        "client": ("testclient", 123),
        "root_path": "",
    }

    with (
        patch("headroom.proxy.server.AnyLLMBackend"),
        patch(
            "headroom.proxy.helpers.apply_session_sticky_ccr_tool",
            side_effect=_inject_proxy_ccr_tool,
        ),
    ):
        app = create_app(config)
        with TestClient(app):
            proxy = app.state.proxy
            proxy_logger = logging.getLogger("headroom.proxy")
            error_records: list[logging.LogRecord] = []
            log_handler = logging.Handler()
            log_handler.setLevel(logging.ERROR)
            log_handler.emit = error_records.append
            proxy_logger.addHandler(log_handler)

            async def delayed_failure(*args, **kwargs):  # noqa: ANN002, ANN003
                started.set()
                await release.wait()
                raise RuntimeError("boom")

            with patch.object(
                proxy.metrics, "record_failed", new_callable=AsyncMock
            ) as record_failed:
                proxy._retry_request = delayed_failure
                task = asyncio.create_task(proxy.handle_anthropic_messages(Request(scope, receive)))
                await started.wait()
                response = await asyncio.wait_for(asyncio.shield(task), 1)
                events: list[dict] = []
                first_body = asyncio.Event()

                async def send(message):  # noqa: ANN001
                    events.append(message)
                    if message["type"] == "http.response.body" and message["body"]:
                        first_body.set()

                response_task = asyncio.create_task(response(scope, receive, send))
                await asyncio.wait_for(first_body.wait(), 2)
                release.set()
                await response_task
                record_failed.assert_awaited_once_with(provider="anthropic")
            proxy_logger.removeHandler(log_handler)

    bodies = [event["body"] for event in events if event["type"] == "http.response.body"]
    assert bodies[0] == b'event: ping\ndata: {"type":"ping"}\n\n'
    assert b"An error occurred while processing the request." in bodies[-1]
    assert b"boom" not in bodies[-1]
    assert events[-1]["more_body"] is False
    assert any(
        record.levelno == logging.ERROR and "RuntimeError: boom" in record.getMessage()
        for record in error_records
    )


@pytest.mark.asyncio
async def test_buffered_ccr_pre_keepalive_exception_returns_json_error() -> None:
    config = _make_config()
    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "stream": True,
        "messages": [{"role": "user", "content": "fail before keepalive"}],
    }

    async def receive():
        return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/messages",
        "raw_path": b"/v1/messages",
        "query_string": b"",
        "headers": [(b"x-api-key", b"test-key"), (b"anthropic-version", b"2023-06-01")],
        "server": ("testserver", 80),
        "client": ("testclient", 123),
        "root_path": "",
    }

    with (
        patch("headroom.proxy.server.AnyLLMBackend"),
        patch(
            "headroom.proxy.helpers.apply_session_sticky_ccr_tool",
            side_effect=_inject_proxy_ccr_tool,
        ),
    ):
        app = create_app(config)
        with TestClient(app):
            proxy = app.state.proxy

            async def early_exception(*args, **kwargs):  # noqa: ANN002, ANN003
                raise RuntimeError("boom")

            proxy._retry_request = early_exception
            response = await proxy.handle_anthropic_messages(Request(scope, receive))
            events: list[dict] = []

            async def send(message):  # noqa: ANN001
                events.append(message)

            await response(scope, receive, send)

    start = next(event for event in events if event["type"] == "http.response.start")
    bodies = [event["body"] for event in events if event["type"] == "http.response.body"]
    assert start["status"] == 502
    assert dict(start["headers"])[b"content-type"] == b"application/json"
    payload = json.loads(bodies[-1].decode())
    assert (
        payload["error"]["message"]
        == "An error occurred while processing your request. Please try again."
    )
