"""Canonical names and matching policy for Headroom MCP tools."""

from __future__ import annotations

import os
from typing import Final

HEADROOM_MCP_TOOL_PREFIX_ENV: Final = "HEADROOM_MCP_TOOL_PREFIX"
DEFAULT_HEADROOM_MCP_TOOL_PREFIX: Final = "mcp__headroom__"

HEADROOM_COMPRESS_TOOL_NAME: Final = "headroom_compress"
HEADROOM_RETRIEVE_TOOL_NAME: Final = "headroom_retrieve"
HEADROOM_STATS_TOOL_NAME: Final = "headroom_stats"
HEADROOM_READ_TOOL_NAME: Final = "headroom_read"

HEADROOM_MCP_TOOL_NAMES: Final[frozenset[str]] = frozenset(
    {
        HEADROOM_COMPRESS_TOOL_NAME,
        HEADROOM_RETRIEVE_TOOL_NAME,
        HEADROOM_STATS_TOOL_NAME,
        HEADROOM_READ_TOOL_NAME,
    }
)


def get_headroom_mcp_tool_prefix() -> str:
    """Return the configured model-visible MCP prefix.

    Claude Code's standalone Headroom MCP server exposes tools as
    ``mcp__headroom__<tool>``. Hosts that use another model-visible spelling
    can set ``HEADROOM_MCP_TOOL_PREFIX`` to the exact replacement prefix.
    """

    configured_prefix = os.environ.get(HEADROOM_MCP_TOOL_PREFIX_ENV, "").strip()
    return configured_prefix or DEFAULT_HEADROOM_MCP_TOOL_PREFIX


def accepted_headroom_mcp_tool_names(
    tool_name: str,
    *,
    prefix: str | None = None,
) -> tuple[str, ...]:
    """Return exact accepted spellings for one canonical Headroom tool."""

    if tool_name not in HEADROOM_MCP_TOOL_NAMES:
        return ()
    resolved_prefix = (
        get_headroom_mcp_tool_prefix()
        if prefix is None
        else prefix.strip() or DEFAULT_HEADROOM_MCP_TOOL_PREFIX
    )
    return (tool_name, f"{resolved_prefix}{tool_name}")


def is_headroom_mcp_tool_name(
    name: object,
    tool_name: str,
    *,
    prefix: str | None = None,
) -> bool:
    """Return whether ``name`` exactly identifies ``tool_name``.

    Bare names remain valid for Headroom's injected tools and raw MCP calls.
    Model-visible names must use the configured full prefix; suffix matching is
    intentionally rejected so unrelated tools cannot impersonate Headroom.
    """

    return isinstance(name, str) and name in accepted_headroom_mcp_tool_names(
        tool_name,
        prefix=prefix,
    )
