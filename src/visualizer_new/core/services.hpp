/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cassert>

namespace lfs::vis {

// Forward declarations
class SceneManager;
class TrainerManager;
class RenderingManager;
class WindowManager;

namespace command {
class CommandHistory;
}

namespace gui {
class GuiManager;
}

/**
 * @brief Service locator for accessing core application services.
 *
 * Provides centralized access to all major managers without requiring
 * pointer-passing chains between components. Services are registered
 * during VisualizerImpl initialization and cleared on shutdown.
 *
 * Usage:
 *   services().scene().selectNode("Model");
 *   services().rendering().markDirty();
 *   services().commands().execute(std::move(cmd));
 *
 * Thread safety: Registration/clear should only happen on main thread.
 * Access is safe from any thread (pointers are stable after init).
 */
class Services {
public:
    static Services& instance() {
        static Services s;
        return s;
    }

    // Registration (called during VisualizerImpl::initialize)
    void set(SceneManager* sm) { scene_manager_ = sm; }
    void set(TrainerManager* tm) { trainer_manager_ = tm; }
    void set(RenderingManager* rm) { rendering_manager_ = rm; }
    void set(WindowManager* wm) { window_manager_ = wm; }
    void set(command::CommandHistory* ch) { command_history_ = ch; }
    void set(gui::GuiManager* gm) { gui_manager_ = gm; }

    // Access - asserts if service not registered
    [[nodiscard]] SceneManager& scene() {
        assert(scene_manager_ && "SceneManager not registered");
        return *scene_manager_;
    }

    [[nodiscard]] TrainerManager& trainer() {
        assert(trainer_manager_ && "TrainerManager not registered");
        return *trainer_manager_;
    }

    [[nodiscard]] RenderingManager& rendering() {
        assert(rendering_manager_ && "RenderingManager not registered");
        return *rendering_manager_;
    }

    [[nodiscard]] WindowManager& window() {
        assert(window_manager_ && "WindowManager not registered");
        return *window_manager_;
    }

    [[nodiscard]] command::CommandHistory& commands() {
        assert(command_history_ && "CommandHistory not registered");
        return *command_history_;
    }

    [[nodiscard]] gui::GuiManager& gui() {
        assert(gui_manager_ && "GuiManager not registered");
        return *gui_manager_;
    }

    // Optional access - returns nullptr if not registered
    [[nodiscard]] SceneManager* sceneOrNull() { return scene_manager_; }
    [[nodiscard]] TrainerManager* trainerOrNull() { return trainer_manager_; }
    [[nodiscard]] RenderingManager* renderingOrNull() { return rendering_manager_; }
    [[nodiscard]] WindowManager* windowOrNull() { return window_manager_; }
    [[nodiscard]] command::CommandHistory* commandsOrNull() { return command_history_; }
    [[nodiscard]] gui::GuiManager* guiOrNull() { return gui_manager_; }

    // Check if all core services are registered
    [[nodiscard]] bool isInitialized() const {
        return scene_manager_ && trainer_manager_ && rendering_manager_ &&
               window_manager_ && command_history_;
    }

    // Clear all services (called during shutdown)
    void clear() {
        scene_manager_ = nullptr;
        trainer_manager_ = nullptr;
        rendering_manager_ = nullptr;
        window_manager_ = nullptr;
        command_history_ = nullptr;
        gui_manager_ = nullptr;
    }

private:
    Services() = default;
    ~Services() = default;
    Services(const Services&) = delete;
    Services& operator=(const Services&) = delete;

    SceneManager* scene_manager_ = nullptr;
    TrainerManager* trainer_manager_ = nullptr;
    RenderingManager* rendering_manager_ = nullptr;
    WindowManager* window_manager_ = nullptr;
    command::CommandHistory* command_history_ = nullptr;
    gui::GuiManager* gui_manager_ = nullptr;
};

// Convenience function
inline Services& services() { return Services::instance(); }

} // namespace lfs::vis
