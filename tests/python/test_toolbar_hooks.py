# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for hook-driven viewport toolbar updates."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


def _install_stub_modules(monkeypatch):
    hook_calls = []
    remove_calls = []

    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(
        add_hook=lambda panel, section, callback, position="append": hook_calls.append(
            (panel, section, callback, position)
        ),
        remove_hook=lambda panel, section, callback: remove_calls.append(
            (panel, section, callback)
        ),
        rml=SimpleNamespace(get_document=lambda _name: None),
    )
    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)

    tools_mod = ModuleType("lfs_plugins.tools")

    class _ToolRegistryStub:
        @staticmethod
        def get_all():
            return []

        @staticmethod
        def get(_tool_id):
            return None

        @staticmethod
        def clear_active():
            return None

        @staticmethod
        def set_active(_tool_id):
            return None

    tools_mod.ToolRegistry = _ToolRegistryStub
    monkeypatch.setitem(sys.modules, "lfs_plugins.tools", tools_mod)

    op_context_mod = ModuleType("lfs_plugins.op_context")
    op_context_mod.get_context = lambda: SimpleNamespace()
    monkeypatch.setitem(sys.modules, "lfs_plugins.op_context", op_context_mod)

    ui_pkg = ModuleType("lfs_plugins.ui")
    ui_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "lfs_plugins.ui", ui_pkg)

    state_mod = ModuleType("lfs_plugins.ui.state")
    state_mod.AppState = SimpleNamespace(trainer_state=SimpleNamespace(value="idle"))
    monkeypatch.setitem(sys.modules, "lfs_plugins.ui.state", state_mod)

    return hook_calls, remove_calls


class _DataModelHandleStub:
    def __init__(self):
        self.dirty_all_calls = 0
        self.dirty_calls = []
        self.record_updates = {}

    def dirty_all(self):
        self.dirty_all_calls += 1

    def dirty(self, name):
        self.dirty_calls.append(name)

    def update_record_list(self, name, records):
        self.record_updates[name] = records


class _DataModelStub:
    def __init__(self):
        self.bound_funcs = {}
        self.bound_events = {}
        self.bound_record_lists = []
        self.handle = _DataModelHandleStub()

    def bind_func(self, name, getter):
        self.bound_funcs[name] = getter

    def bind_event(self, name, callback):
        self.bound_events[name] = callback

    def bind_record_list(self, name):
        self.bound_record_lists.append(name)

    def get_handle(self):
        return self.handle


@pytest.fixture
def toolbar_module(monkeypatch):
    project_root = Path(__file__).parent.parent.parent
    source_python = project_root / "src" / "python"
    if str(source_python) in sys.path:
        sys.path.remove(str(source_python))
    sys.path.insert(0, str(source_python))

    sys.modules.pop("lfs_plugins", None)
    sys.modules.pop("lfs_plugins.toolbar", None)
    hook_calls, remove_calls = _install_stub_modules(monkeypatch)
    module = import_module("lfs_plugins.toolbar")
    return module, hook_calls, remove_calls


def test_toolbar_binds_overlay_model_fields(toolbar_module):
    module, _hook_calls, _remove_calls = toolbar_module
    model = _DataModelStub()

    module.reset_overlay_state()
    module.bind_overlay_model(model)

    assert "show_render_controls" in model.bound_funcs
    assert "camera_flyout_open" in model.bound_funcs
    assert "render_flyout_open" in model.bound_funcs
    assert "camera_group_buttons" in model.bound_record_lists
    assert "render_group_buttons" in model.bound_record_lists
    assert "camera_mode_buttons" in model.bound_record_lists
    assert "render_mode_buttons" in model.bound_record_lists
    assert "toolbar_action" in model.bound_events


def test_toolbar_attach_handle_marks_model_dirty(toolbar_module):
    module, _hook_calls, _remove_calls = toolbar_module
    handle = _DataModelHandleStub()

    module.reset_overlay_state()
    module.attach_overlay_model_handle(handle)

    assert handle.dirty_all_calls == 1


def test_utility_group_button_preserves_active_state(toolbar_module):
    module, _hook_calls, _remove_calls = toolbar_module

    inactive = module._button_record(
        "util-camera-orbit",
        "set_camera_navigation_mode",
        "orbit",
        "../icon/camera-orbit.png",
        tooltip_text="Orbit Camera",
        selected=False,
    )
    active = module._button_record(
        "util-camera-trackball",
        "set_camera_navigation_mode",
        "trackball",
        "../icon/world.png",
        tooltip_text="Free Orbit Camera",
        selected=True,
    )

    group_button = module._UtilityToolbarController._group_button(
        "camera",
        [inactive, active],
        "Camera Mode",
    )[0]

    assert group_button["button_id"] == "group-camera"
    assert group_button["action"] == "toggle_flyout"
    assert group_button["value"] == "camera"
    assert group_button["icon_src"] == "../icon/world.png"
    assert group_button["selected"] is True


def test_viewport_overlay_flyout_template_uses_one_expanded_bar():
    project_root = Path(__file__).parent.parent.parent
    rml_path = (
        project_root
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "resources"
        / "viewport_overlay.rml"
    )
    rcss_path = rml_path.with_suffix(".rcss")
    rml = rml_path.read_text(encoding="utf-8")
    rcss = rcss_path.read_text(encoding="utf-8")

    assert "toolbar-flyout-anchor" not in rml
    assert rml.count('class="toolbar-flyout toolbar-flyout-camera hidden"') == 2
    assert rml.count('class="toolbar-flyout toolbar-flyout-render hidden"') == 2
    assert rml.count('<span class="flyout-corner-marker"></span>') == 4
    assert "dropdown-arrow.png" not in rml
    assert 'data-class-hidden="camera_flyout_open"' not in rml
    assert 'data-class-hidden="render_flyout_open"' not in rml
    assert 'data-for="button : camera_mode_buttons"' in rml
    assert 'data-for="button : render_mode_buttons"' in rml
    assert 'data-class-selected="button.selected"' in rml
    assert rml.count('data-class-hidden="!camera_flyout_open"') == 2
    assert rml.count('data-class-hidden="!render_flyout_open"') == 2
    assert "toolbar-flyout-camera" in rcss
    assert "width: 96dp;" in rcss
    assert "toolbar-flyout-render" in rcss
    assert "width: 128dp;" in rcss
    assert "left: 100%;" in rcss
    assert ".toolbar-flyout-trigger.hidden" not in rcss
    assert "right: 0;" in rcss
    assert "bottom: 0;" in rcss
    assert "width: 0;" in rcss
    assert "height: 0;" in rcss
    assert "border-left-color: rgba(0, 0, 0, 0);" in rcss
    assert "border-bottom-width: 9dp;" in rcss


def test_viewport_toolbar_update_syncs_flyout_state(toolbar_module, monkeypatch):
    module, _hook_calls, _remove_calls = toolbar_module
    model = _DataModelStub()
    lf_stub = sys.modules["lichtfeld"]

    lf_stub.RenderMode = SimpleNamespace(
        SPLATS="splats",
        POINTS="points",
        RINGS="rings",
        CENTERS="centers",
    )
    lf_stub.get_camera_navigation_mode = lambda: "orbit"
    lf_stub.get_camera_view_snap_enabled = lambda: False
    lf_stub.get_render_mode = lambda: lf_stub.RenderMode.SPLATS
    lf_stub.is_fullscreen = lambda: False
    lf_stub.is_orthographic = lambda: False
    lf_stub.get_depth_view = lambda: False
    monkeypatch.setattr(lf_stub.ui, "context", lambda: SimpleNamespace(), raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_active_tool", lambda: "", raising=False)
    monkeypatch.setattr(lf_stub.ui, "get_split_view_mode", lambda: "single", raising=False)
    monkeypatch.setattr(lf_stub.ui, "is_sequencer_visible", lambda: False, raising=False)
    monkeypatch.setattr(lf_stub.ui, "is_panel_enabled", lambda _panel_id: False, raising=False)
    monkeypatch.setattr(module, "histogram_mode_available", lambda _context: False)

    module.reset_overlay_state()
    module.bind_overlay_model(model)
    module.attach_overlay_model_handle(model.handle)
    model.handle.record_updates.clear()

    module.update_overlay(SimpleNamespace())

    camera_buttons = model.handle.record_updates["camera_mode_buttons"]
    camera_group = model.handle.record_updates["camera_group_buttons"][0]
    assert len(camera_buttons) == 3
    assert camera_group["icon_src"] == "../icon/camera-orbit.png"
    assert camera_group["selected"] is True
    assert model.bound_funcs["camera_flyout_open"]() is False

    model.bound_events["toolbar_action"](None, None, ["toggle_flyout", "camera"])
    module.update_overlay(SimpleNamespace())

    assert model.bound_funcs["camera_flyout_open"]() is True
    assert "camera_flyout_open" in model.handle.dirty_calls
