# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Toolbar tools using declarative ToolDef system."""

from typing import Optional

import lichtfeld as lf

from .tool_defs.builtin import BUILTIN_TOOLS, get_tool_by_id
from .tool_defs.definition import ToolDef


class ToolRegistry:
    """Manages tool activation state."""

    _active_tool_id: str = ""

    @classmethod
    def get(cls, tool_id: str) -> Optional[ToolDef]:
        """Get tool definition by ID."""
        return get_tool_by_id(tool_id)

    @classmethod
    def get_all(cls) -> list[ToolDef]:
        """Get all tool definitions."""
        return list(BUILTIN_TOOLS)

    @classmethod
    def set_active(cls, tool_id: str) -> bool:
        """Set the active tool by ID."""
        tool = get_tool_by_id(tool_id)
        if not tool:
            return False

        from .op_context import get_context

        context = get_context()
        if not tool.can_activate(context):
            return False

        lf.ui.ops.cancel_modal()
        lf.ui.clear_gizmo()

        cls._active_tool_id = tool_id

        gizmo = tool.gizmo or ""
        lf.ui.set_active_operator(tool_id, gizmo)

        if tool_id == "builtin.select" and not lf.ui.get_active_submode():
            lf.ui.set_selection_mode("centers")

        if tool.gizmo and not tool.operator:
            lf.ui.set_gizmo_type(tool.gizmo)
        elif tool.operator:
            lf.ui.ops.invoke(tool.operator)

        return True

    @classmethod
    def get_active(cls) -> Optional[ToolDef]:
        """Get the active tool definition."""
        return get_tool_by_id(cls._active_tool_id)

    @classmethod
    def get_active_id(cls) -> str:
        """Get the active tool ID."""
        return cls._active_tool_id

    @classmethod
    def clear(cls):
        """Clear active tool state."""
        cls._active_tool_id = ""


def register():
    """Initialize tools system."""
    if BUILTIN_TOOLS:
        ToolRegistry.set_active("builtin.select")


def unregister():
    """Cleanup tools system."""
    ToolRegistry.clear()
