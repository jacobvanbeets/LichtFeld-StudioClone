# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tool manager for activation and state.

The ToolManager handles tool activation, tracks the active tool, and
manages the transition between tools.
"""

from __future__ import annotations

from typing import Any

import lichtfeld as lf

from .definition import ToolDef
from .builtin import BUILTIN_TOOLS, get_tool_by_id


class ToolManager:
    """Manages tool activation and state.

    This is a singleton that tracks which tool is active and handles
    tool transitions.
    """

    _instance: ToolManager | None = None
    _active_tool_id: str = ""
    _custom_tools: dict[str, ToolDef] = {}

    @classmethod
    def instance(cls) -> ToolManager:
        if cls._instance is None:
            cls._instance = ToolManager()
        return cls._instance

    def get_active_id(self) -> str:
        """Get the ID of the currently active tool."""
        return self._active_tool_id

    def get_active(self) -> ToolDef | None:
        """Get the currently active tool definition."""
        return self.get_tool(self._active_tool_id)

    def get_tool(self, tool_id: str) -> ToolDef | None:
        """Get a tool by ID from builtins or custom tools."""
        tool = get_tool_by_id(tool_id)
        if tool:
            return tool
        return self._custom_tools.get(tool_id)

    def get_all_tools(self) -> list[ToolDef]:
        """Get all tools (builtin + custom), sorted by group then order."""
        all_tools = list(BUILTIN_TOOLS) + list(self._custom_tools.values())
        return sorted(all_tools, key=lambda t: (t.group, t.order))

    def register_tool(self, tool: ToolDef) -> None:
        """Register a custom tool.

        Args:
            tool: Tool definition to register.
        """
        self._custom_tools[tool.id] = tool

    def unregister_tool(self, tool_id: str) -> None:
        """Unregister a custom tool.

        Args:
            tool_id: ID of tool to unregister.
        """
        self._custom_tools.pop(tool_id, None)
        if self._active_tool_id == tool_id:
            self._active_tool_id = ""

    def set_active(self, tool_id: str, context: Any = None) -> bool:
        """Set the active tool.

        Args:
            tool_id: ID of tool to activate.
            context: Optional context for poll check.

        Returns:
            True if tool was activated successfully.
        """
        tool = self.get_tool(tool_id)
        if not tool:
            return False

        if context and not tool.can_activate(context):
            return False

        lf.ui.ops.cancel_modal()
        lf.ui.clear_gizmo()

        self._active_tool_id = tool_id

        gizmo = tool.gizmo or ""
        lf.ui.set_active_operator(tool_id, gizmo)

        if tool_id == "builtin.select" and not lf.ui.get_active_submode():
            lf.ui.set_selection_mode("centers")

        if tool.gizmo and not tool.operator:
            lf.ui.set_gizmo_type(tool.gizmo)
        elif tool.operator:
            lf.ui.ops.invoke(tool.operator)

        return True

    def poll_tool(self, tool_id: str, context: Any) -> bool:
        """Check if a tool can be activated.

        Args:
            tool_id: ID of tool to check.
            context: Context for poll check.

        Returns:
            True if tool can be activated.
        """
        tool = self.get_tool(tool_id)
        if not tool:
            return False
        return tool.can_activate(context)

    def reset(self) -> None:
        """Reset the manager state."""
        self._active_tool_id = ""
        self._custom_tools.clear()


def get_active_tool() -> ToolDef | None:
    """Convenience function to get the active tool."""
    return ToolManager.instance().get_active()


def set_active_tool(tool_id: str, context: Any = None) -> bool:
    """Convenience function to set the active tool."""
    return ToolManager.instance().set_active(tool_id, context)
