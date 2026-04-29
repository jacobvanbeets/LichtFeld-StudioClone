/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

struct SDL_Window;
class RenderInterface_VK;

namespace Rml {
    class Context;
    class RenderInterface;
}

namespace lfs::vis {
    class VulkanContext;
}

namespace lfs::vis::gui {

    class RmlFBO;
    class RmlSystemInterface;
    class RmlRenderInterface;
    class RmlTextInputHandler;
    enum class RmlCursorRequest : uint8_t;

    class RmlUIManager {
    public:
        RmlUIManager();
        ~RmlUIManager();

        bool init(SDL_Window* window, float dp_ratio = 1.0f);
        bool initVulkan(SDL_Window* window, lfs::vis::VulkanContext& vulkan_context, float dp_ratio = 1.0f);
        void shutdown();
        [[nodiscard]] bool isInitialized() const { return initialized_; }

        float getDpRatio() const { return dp_ratio_; }
        void setDpRatio(float ratio);

        Rml::Context* createContext(const std::string& name, int width, int height);
        Rml::Context* getContext(const std::string& name);
        void destroyContext(const std::string& name);

        void setResizeDeferring(bool defer) { resize_deferring_ = defer; }
        [[nodiscard]] bool shouldDeferFboUpdate(const RmlFBO& fbo) const;

        void activateTheme(const std::string& theme_id);
        const std::string& activeThemeId() const { return active_theme_id_; }

        RmlRenderInterface* getRenderInterface() const { return render_interface_; }
        RenderInterface_VK* getVulkanRenderInterface() const { return vulkan_render_interface_; }
        RmlTextInputHandler* getTextInputHandler() const { return text_input_handler_.get(); }
        SDL_Window* getWindow() const { return window_; }

        void beginFrameCursorTracking();
        void trackContextFrame(const Rml::Context* context, int window_x, int window_y);
        RmlCursorRequest consumeCursorRequest();

    private:
        bool initWithRenderInterface(SDL_Window* window,
                                     float dp_ratio,
                                     std::unique_ptr<Rml::RenderInterface> render_interface,
                                     RmlRenderInterface* gl_render_interface,
                                     RenderInterface_VK* vulkan_render_interface);

        std::unique_ptr<RmlSystemInterface> system_interface_;
        std::unique_ptr<Rml::RenderInterface> owned_render_interface_;
        RmlRenderInterface* render_interface_ = nullptr;
        RenderInterface_VK* vulkan_render_interface_ = nullptr;
        std::unique_ptr<RmlTextInputHandler> text_input_handler_;
        std::unordered_map<std::string, Rml::Context*> contexts_;
        SDL_Window* window_ = nullptr;
        float dp_ratio_ = 1.0f;
        std::string active_theme_id_;
        bool resize_deferring_ = false;
        bool debugger_enabled_ = false;
        bool debugger_initialized_ = false;
        bool initialized_ = false;
    };

} // namespace lfs::vis::gui
