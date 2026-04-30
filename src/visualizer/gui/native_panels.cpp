/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/native_panels.hpp"
#include "gui/gizmo_manager.hpp"
#include "gui/gui_manager.hpp"
#include "gui/panel_layout.hpp"
#include "gui/panel_registry.hpp"
#include "gui/rml_status_bar.hpp"
#include "gui/sequencer_ui_manager.hpp"
#include "gui/startup_overlay.hpp"
#include "core/camera.hpp"
#include "core/scene.hpp"
#include "internal/viewport.hpp"
#include "python/python_runtime.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/rendering_manager.hpp"
#include "theme/theme.hpp"
#include "visualizer/gui/video_widget_interface.hpp"
#include "visualizer_impl.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/constants.hpp>
#include <imgui.h>

namespace lfs::vis::gui::native_panels {

    namespace {
        struct GuidePanelTarget {
            SplitViewPanelId panel = SplitViewPanelId::Left;
            const Viewport* viewport = nullptr;
            glm::vec2 pos{0.0f};
            glm::vec2 size{0.0f};
            glm::ivec2 render_size{0};
            ClipRect clip_rect{};

            [[nodiscard]] bool valid() const {
                return viewport != nullptr && size.x > 0.0f && size.y > 0.0f &&
                       render_size.x > 0 && render_size.y > 0 &&
                       clip_rect.width > 0 && clip_rect.height > 0;
            }
        };

        [[nodiscard]] glm::vec4 toGuideColor(const glm::vec3& color, const float alpha) {
            return {color.r, color.g, color.b, alpha};
        }

        [[nodiscard]] glm::vec4 toGuideColor(const ImVec4& color, const float alpha) {
            return {color.x, color.y, color.z, alpha};
        }

        void addResolvedPanel(std::vector<GuidePanelTarget>& panels,
                              const std::optional<RenderingManager::ViewerPanelInfo>& info_opt) {
            if (!info_opt || !info_opt->valid()) {
                return;
            }
            const auto& info = *info_opt;
            panels.push_back(GuidePanelTarget{
                .panel = info.panel,
                .viewport = info.viewport,
                .pos = {info.x, info.y},
                .size = {info.width, info.height},
                .render_size = {info.render_width, info.render_height},
                .clip_rect =
                    {.x = static_cast<int>(std::round(info.x)),
                     .y = static_cast<int>(std::round(info.y)),
                     .width = static_cast<int>(std::round(info.width)),
                     .height = static_cast<int>(std::round(info.height))},
            });
        }

        [[nodiscard]] std::vector<GuidePanelTarget> collectGuidePanels(
            const VisualizerImpl& viewer,
            const ViewportLayout& viewport_layout,
            const RenderingManager& rendering_manager) {
            std::vector<GuidePanelTarget> panels;
            panels.reserve(2);

            const auto& viewport = viewer.getViewport();
            if (rendering_manager.isIndependentSplitViewActive()) {
                addResolvedPanel(panels, rendering_manager.resolveViewerPanel(
                                             viewport, viewport_layout.pos, viewport_layout.size,
                                             std::nullopt, SplitViewPanelId::Left));
                addResolvedPanel(panels, rendering_manager.resolveViewerPanel(
                                             viewport, viewport_layout.pos, viewport_layout.size,
                                             std::nullopt, SplitViewPanelId::Right));
            }

            if (!panels.empty()) {
                return panels;
            }

            const int clip_x = static_cast<int>(std::round(viewport_layout.pos.x));
            const int clip_y = static_cast<int>(std::round(viewport_layout.pos.y));
            const int clip_w = static_cast<int>(std::round(viewport_layout.size.x));
            const int clip_h = static_cast<int>(std::round(viewport_layout.size.y));
            std::vector<ClipRect> clips;
            clips.reserve(2);
            if (const auto divider = rendering_manager.getSplitDividerScreenX(
                    viewport_layout.pos, viewport_layout.size)) {
                const int divider_x = std::clamp(static_cast<int>(std::round(*divider)), clip_x, clip_x + clip_w);
                if (divider_x > clip_x) {
                    clips.push_back({clip_x, clip_y, divider_x - clip_x, clip_h});
                }
                if (divider_x < clip_x + clip_w) {
                    clips.push_back({divider_x, clip_y, clip_x + clip_w - divider_x, clip_h});
                }
            }
            if (clips.empty()) {
                clips.push_back({clip_x, clip_y, clip_w, clip_h});
            }

            const glm::ivec2 render_size(
                std::max(static_cast<int>(std::round(viewport_layout.size.x)), 1),
                std::max(static_cast<int>(std::round(viewport_layout.size.y)), 1));
            for (size_t i = 0; i < clips.size(); ++i) {
                panels.push_back(GuidePanelTarget{
                    .panel = i == 0 ? SplitViewPanelId::Left : SplitViewPanelId::Right,
                    .viewport = &viewport,
                    .pos = viewport_layout.pos,
                    .size = viewport_layout.size,
                    .render_size = render_size,
                    .clip_rect = clips[i],
                });
            }
            return panels;
        }

        [[nodiscard]] std::optional<glm::vec2> projectToScreen(
            const GuidePanelTarget& panel,
            const RenderSettings& settings,
            const glm::vec3& world_point) {
            const auto projected = lfs::rendering::projectWorldPoint(
                panel.viewport->getRotationMatrix(),
                panel.viewport->getTranslation(),
                panel.render_size,
                world_point,
                settings.focal_length_mm,
                settings.orthographic,
                settings.ortho_scale);
            if (!projected) {
                return std::nullopt;
            }
            const float sx = panel.size.x / static_cast<float>(std::max(panel.render_size.x, 1));
            const float sy = panel.size.y / static_cast<float>(std::max(panel.render_size.y, 1));
            return glm::vec2(panel.pos.x + projected->x * sx, panel.pos.y + projected->y * sy);
        }

        void addProjectedLine(LineRenderer& lines,
                              const GuidePanelTarget& panel,
                              const RenderSettings& settings,
                              const glm::vec3& a,
                              const glm::vec3& b,
                              const glm::vec4& color,
                              const float thickness) {
            const auto pa = projectToScreen(panel, settings, a);
            const auto pb = projectToScreen(panel, settings, b);
            if (pa && pb) {
                lines.addLine(*pa, *pb, color, thickness);
            }
        }

        [[nodiscard]] std::optional<glm::mat4> cameraVisualizerTransform(
            const lfs::core::Camera& camera,
            const glm::mat4& scene_transform) {
            auto rotation_tensor = camera.R();
            auto translation_tensor = camera.T();
            if (!rotation_tensor.is_valid() || !translation_tensor.is_valid()) {
                return std::nullopt;
            }
            if (rotation_tensor.device() != lfs::core::Device::CPU) {
                rotation_tensor = rotation_tensor.cpu();
            }
            if (translation_tensor.device() != lfs::core::Device::CPU) {
                translation_tensor = translation_tensor.cpu();
            }
            if (rotation_tensor.dtype() != lfs::core::DataType::Float32 ||
                translation_tensor.dtype() != lfs::core::DataType::Float32 ||
                rotation_tensor.numel() < 9 || translation_tensor.numel() < 3) {
                return std::nullopt;
            }

            const float* const rotation = rotation_tensor.ptr<float>();
            const float* const translation = translation_tensor.ptr<float>();
            if (!rotation || !translation) {
                return std::nullopt;
            }
            glm::mat4 world_to_camera(1.0f);
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    world_to_camera[col][row] = rotation[row * 3 + col];
                }
                world_to_camera[3][row] = translation[row];
            }
            return scene_transform * glm::inverse(world_to_camera) *
                   lfs::rendering::DATA_TO_VISUALIZER_CAMERA_AXES_4;
        }

        void drawGrid(LineRenderer& lines,
                      const GuidePanelTarget& panel,
                      const RenderingManager& rendering_manager,
                      const RenderSettings& settings) {
            if (!settings.show_grid || settings.grid_opacity <= 0.0f) {
                return;
            }

            const int plane = rendering_manager.getGridPlaneForPanel(panel.panel);
            const glm::vec3 camera_pos = panel.viewport->getTranslation();
            const float distance = glm::length(camera_pos - panel.viewport->camera.getPivot());
            const float ortho_extent = settings.orthographic
                                           ? std::max(panel.size.x, panel.size.y) /
                                                 std::max(settings.ortho_scale, 1e-3f)
                                           : 0.0f;
            const float extent = std::clamp(std::max({10.0f, distance * 2.5f, ortho_extent * 1.5f}),
                                            1.0f, 100000.0f);
            float step = std::pow(10.0f, std::floor(std::log10(std::max(extent / 20.0f, 1e-3f))));
            if (extent / step > 40.0f) {
                step *= 2.0f;
            }
            const int line_count = std::clamp(static_cast<int>(std::ceil(extent / step)), 8, 80);
            const auto& t = theme();
            const glm::vec4 regular = toGuideColor(t.palette.text_dim, settings.grid_opacity * 0.22f);
            const glm::vec4 major = toGuideColor(t.palette.text_dim, settings.grid_opacity * 0.42f);
            constexpr glm::vec4 X_AXIS(0.9f, 0.22f, 0.22f, 0.72f);
            constexpr glm::vec4 Y_AXIS(0.25f, 0.75f, 0.25f, 0.72f);
            constexpr glm::vec4 Z_AXIS(0.28f, 0.48f, 0.95f, 0.72f);

            const auto make_point = [plane](const float a, const float b) {
                switch (plane) {
                case 0: return glm::vec3(0.0f, a, b); // YZ
                case 2: return glm::vec3(a, b, 0.0f); // XY
                case 1:
                default: return glm::vec3(a, 0.0f, b); // XZ
                }
            };

            const glm::vec3 pivot = panel.viewport->camera.getPivot();
            const float center_a = plane == 0 ? pivot.y : pivot.x;
            const float center_b = plane == 2 ? pivot.y : pivot.z;
            const float start_a = std::floor(center_a / step) * step - static_cast<float>(line_count) * step;
            const float start_b = std::floor(center_b / step) * step - static_cast<float>(line_count) * step;
            const float min_a = start_a;
            const float max_a = start_a + static_cast<float>(line_count * 2) * step;
            const float min_b = start_b;
            const float max_b = start_b + static_cast<float>(line_count * 2) * step;

            for (int i = 0; i <= line_count * 2; ++i) {
                const float a = start_a + static_cast<float>(i) * step;
                const bool axis = std::abs(a) <= step * 0.25f;
                const bool is_major = i % 10 == 0;
                glm::vec4 color = axis ? (plane == 0 ? Y_AXIS : X_AXIS) : (is_major ? major : regular);
                addProjectedLine(lines, panel, settings, make_point(a, min_b), make_point(a, max_b),
                                 color, axis ? 1.4f : 1.0f);

                const float b = start_b + static_cast<float>(i) * step;
                const bool b_axis = std::abs(b) <= step * 0.25f;
                const bool b_major = i % 10 == 0;
                color = b_axis ? (plane == 2 ? Y_AXIS : Z_AXIS) : (b_major ? major : regular);
                addProjectedLine(lines, panel, settings, make_point(min_a, b), make_point(max_a, b),
                                 color, b_axis ? 1.4f : 1.0f);
            }
        }

        void drawAxesAndPivot(LineRenderer& lines,
                              const GuidePanelTarget& panel,
                              const RenderSettings& settings) {
            if (settings.show_coord_axes) {
                constexpr std::array AXIS_COLORS{
                    glm::vec4(0.95f, 0.18f, 0.18f, 0.9f),
                    glm::vec4(0.25f, 0.82f, 0.25f, 0.9f),
                    glm::vec4(0.30f, 0.50f, 1.0f, 0.9f),
                };
                constexpr std::array AXES{
                    glm::vec3(1.0f, 0.0f, 0.0f),
                    glm::vec3(0.0f, 1.0f, 0.0f),
                    glm::vec3(0.0f, 0.0f, 1.0f),
                };
                for (size_t axis = 0; axis < AXES.size(); ++axis) {
                    if (settings.axes_visibility[axis]) {
                        addProjectedLine(lines, panel, settings, glm::vec3(0.0f),
                                         AXES[axis] * settings.axes_size,
                                         AXIS_COLORS[axis], 2.0f);
                    }
                }
            }

            if (settings.show_pivot) {
                const auto pivot = projectToScreen(panel, settings, panel.viewport->camera.getPivot());
                if (pivot) {
                    const auto& t = theme();
                    lines.addCircleFilled(*pivot, 4.0f, toGuideColor(t.palette.warning, 0.9f), 16);
                    lines.addLine(*pivot + glm::vec2(-8.0f, 0.0f), *pivot + glm::vec2(8.0f, 0.0f),
                                  toGuideColor(t.palette.warning, 0.7f), 1.2f);
                    lines.addLine(*pivot + glm::vec2(0.0f, -8.0f), *pivot + glm::vec2(0.0f, 8.0f),
                                  toGuideColor(t.palette.warning, 0.7f), 1.2f);
                }
            }
        }

        void drawCameraFrustums(LineRenderer& lines,
                                const GuidePanelTarget& panel,
                                const RenderingManager& rendering_manager,
                                const RenderSettings& settings,
                                lfs::core::Scene& scene) {
            if (!settings.show_camera_frustums || settings.camera_frustum_scale <= 0.0f) {
                return;
            }

            const auto cameras = scene.getVisibleCameras();
            if (cameras.empty()) {
                return;
            }
            auto scene_transforms = scene.getVisibleCameraSceneTransforms();
            for (auto& transform : scene_transforms) {
                transform = lfs::rendering::dataWorldTransformToVisualizerWorld(transform);
            }

            const auto disabled_uids = scene.getTrainingDisabledCameraUids();
            const int current_camera_uid = rendering_manager.getCurrentCameraId();
            constexpr std::array<std::pair<int, int>, 8> EDGES{{
                {0, 1}, {0, 2}, {0, 3}, {0, 4},
                {1, 2}, {2, 3}, {3, 4}, {4, 1},
            }};

            for (size_t i = 0; i < cameras.size(); ++i) {
                const auto& camera = cameras[i];
                if (!camera) {
                    continue;
                }
                const int image_width = camera->image_width() > 0 ? camera->image_width() : camera->camera_width();
                const int image_height = camera->image_height() > 0 ? camera->image_height() : camera->camera_height();
                if (image_width <= 0 || image_height <= 0) {
                    continue;
                }
                glm::mat4 scene_transform(1.0f);
                if (i < scene_transforms.size()) {
                    scene_transform = scene_transforms[i];
                }
                const auto visualizer_c2w = cameraVisualizerTransform(*camera, scene_transform);
                if (!visualizer_c2w) {
                    continue;
                }

                glm::vec4 color = toGuideColor(
                    camera->split() == lfs::core::CameraSplit::Eval || camera->image_name().find("test") != std::string::npos
                        ? settings.eval_camera_color
                        : settings.train_camera_color,
                    disabled_uids.count(camera->uid()) > 0 ? 0.28f : 0.72f);
                float thickness = 1.4f;
                if (camera->uid() == current_camera_uid) {
                    color.a = 0.95f;
                    thickness = 2.4f;
                }

                const bool equirectangular =
                    camera->camera_model_type() == lfs::core::CameraModelType::EQUIRECTANGULAR;
                if (equirectangular) {
                    constexpr int SEGMENTS = 48;
                    for (int circle = 0; circle < 3; ++circle) {
                        std::optional<glm::vec3> previous_world;
                        std::optional<glm::vec3> first_world;
                        for (int segment = 0; segment <= SEGMENTS; ++segment) {
                            const float a = static_cast<float>(segment % SEGMENTS) /
                                            static_cast<float>(SEGMENTS) * 2.0f * glm::pi<float>();
                            glm::vec3 local(0.0f);
                            if (circle == 0) {
                                local = {std::cos(a), std::sin(a), 0.0f};
                            } else if (circle == 1) {
                                local = {std::cos(a), 0.0f, std::sin(a)};
                            } else {
                                local = {0.0f, std::cos(a), std::sin(a)};
                            }
                            const glm::vec3 world = glm::vec3(
                                *visualizer_c2w * glm::vec4(local * settings.camera_frustum_scale, 1.0f));
                            if (!first_world) {
                                first_world = world;
                            }
                            if (previous_world) {
                                addProjectedLine(lines, panel, settings, *previous_world, world, color, thickness);
                            }
                            previous_world = world;
                        }
                        (void)first_world;
                    }
                    continue;
                }

                if (camera->focal_y() <= 0.0f) {
                    continue;
                }
                const float aspect = static_cast<float>(image_width) / static_cast<float>(image_height);
                const float fov_y = lfs::core::focal2fov(camera->focal_y(), image_height);
                const float depth = settings.camera_frustum_scale;
                const float half_height = std::tan(fov_y * 0.5f) * depth;
                const float half_width = half_height * aspect;
                const std::array local_points{
                    glm::vec3(0.0f, 0.0f, 0.0f),
                    glm::vec3(-half_width, half_height, -depth),
                    glm::vec3(half_width, half_height, -depth),
                    glm::vec3(half_width, -half_height, -depth),
                    glm::vec3(-half_width, -half_height, -depth),
                };
                std::array<glm::vec3, 5> world_points{};
                for (size_t p = 0; p < local_points.size(); ++p) {
                    world_points[p] = glm::vec3(*visualizer_c2w * glm::vec4(local_points[p], 1.0f));
                }
                for (const auto& [a, b] : EDGES) {
                    addProjectedLine(lines, panel, settings,
                                     world_points[static_cast<size_t>(a)],
                                     world_points[static_cast<size_t>(b)],
                                     color, thickness);
                }
            }
        }
    } // namespace

    VideoExtractorPanel::VideoExtractorPanel(lfs::gui::IVideoExtractorWidget* widget)
        : widget_(widget) {}

    void VideoExtractorPanel::draw(const PanelDrawContext& ctx) {
        (void)ctx;
        if (!widget_ || !widget_->render())
            PanelRegistry::instance().set_panel_enabled("native.video_extractor", false);
    }

    StartupOverlayPanel::StartupOverlayPanel(StartupOverlay* overlay, const bool* drag_hovering)
        : overlay_(overlay),
          drag_hovering_(drag_hovering) {}

    void StartupOverlayPanel::draw(const PanelDrawContext& ctx) {
        if (ctx.viewport)
            overlay_->render(*ctx.viewport, drag_hovering_ ? *drag_hovering_ : false);
    }

    bool StartupOverlayPanel::poll(const PanelDrawContext& ctx) {
        (void)ctx;
        return overlay_->isVisible();
    }

    SelectionOverlayPanel::SelectionOverlayPanel(GuiManager* gui)
        : gui_(gui) {}

    void SelectionOverlayPanel::draw(const PanelDrawContext& ctx) {
        if (ctx.ui)
            gui_->renderSelectionOverlays(*ctx.ui);
    }

    ViewportDecorationsPanel::ViewportDecorationsPanel(GuiManager* gui)
        : gui_(gui) {}

    void ViewportDecorationsPanel::draw(const PanelDrawContext& ctx) {
        (void)ctx;
        gui_->renderViewportDecorations();
    }

    void ViewportSceneGuidesPanel::draw(const PanelDrawContext& ctx) {
        if (ctx.ui_hidden || !ctx.ui || !ctx.ui->viewer || !ctx.viewport || !ctx.scene) {
            return;
        }

        auto* const rendering_manager = ctx.ui->viewer->getRenderingManager();
        auto* const window_manager = ctx.ui->viewer->getWindowManager();
        if (!rendering_manager || !window_manager ||
            ctx.viewport->size.x <= 0.0f || ctx.viewport->size.y <= 0.0f) {
            return;
        }

        const RenderSettings settings = rendering_manager->getSettings();
        if (!settings.show_grid && !settings.show_coord_axes && !settings.show_pivot &&
            !settings.show_camera_frustums) {
            return;
        }

        const auto panels = collectGuidePanels(*ctx.ui->viewer, *ctx.viewport, *rendering_manager);
        if (panels.empty()) {
            return;
        }

        const glm::ivec2 screen_size = window_manager->getWindowSize();
        const glm::ivec2 framebuffer_size = window_manager->getFramebufferSize();
        for (const auto& panel : panels) {
            if (!panel.valid()) {
                continue;
            }
            line_renderer_.begin(screen_size.x, screen_size.y,
                                 framebuffer_size.x, framebuffer_size.y,
                                 panel.clip_rect);
            drawGrid(line_renderer_, panel, *rendering_manager, settings);
            drawAxesAndPivot(line_renderer_, panel, settings);
            drawCameraFrustums(line_renderer_, panel, *rendering_manager, settings, *ctx.scene);
            line_renderer_.end();
        }
    }

    bool ViewportSceneGuidesPanel::poll(const PanelDrawContext& ctx) {
        return !ctx.ui_hidden && ctx.viewport &&
               ctx.viewport->size.x > 0.0f && ctx.viewport->size.y > 0.0f &&
               ctx.scene != nullptr;
    }

    SequencerPanel::SequencerPanel(SequencerUIManager* seq, const PanelLayoutManager* layout)
        : seq_(seq),
          layout_(layout) {}

    void SequencerPanel::draw(const PanelDrawContext& ctx) {
        (void)ctx;
    }

    void SequencerPanel::preloadDirect(const float w, const float h,
                                       const PanelDrawContext& ctx,
                                       const float clip_y_min,
                                       const float clip_y_max,
                                       const PanelInputState* input) {
        (void)w;
        (void)ctx;
        (void)clip_y_min;
        (void)clip_y_max;
        input_ = input;

        if (seq_)
            seq_->setFloating(is_floating_);

        if (is_floating_) {
            const float preferred_h = seq_ ? seq_->preferredFloatingHeight() : 0.0f;
            direct_draw_height_ = forced_height_ > 0.0f
                                      ? forced_height_
                                      : std::min(h, std::max(0.0f, preferred_h));
        } else {
            direct_draw_height_ = h;
        }
    }

    void SequencerPanel::drawDirect(const float x, const float y,
                                    const float w, const float h,
                                    const PanelDrawContext& ctx) {
        if (seq_)
            seq_->setFloating(is_floating_);

        if (is_floating_) {
            direct_draw_height_ = seq_ ? std::max(0.0f, seq_->preferredFloatingHeight()) : h;
        } else {
            direct_draw_height_ = h;
        }

        if (seq_ && ctx.ui && ctx.viewport && input_ && h > 0.0f)
            seq_->render(*ctx.ui, *ctx.viewport, x, y, w, h, *input_);
    }

    bool SequencerPanel::poll(const PanelDrawContext& ctx) {
        const bool is_enabled = !ctx.ui_hidden && ctx.ui && ctx.ui->editor &&
                                !ctx.ui->editor->isToolsDisabled() && layout_->isShowSequencer();
        if (!is_enabled && seq_)
            seq_->setSequencerEnabled(false);
        return is_enabled;
    }

    NodeTransformGizmoPanel::NodeTransformGizmoPanel(GizmoManager* gizmo)
        : gizmo_(gizmo) {}

    void NodeTransformGizmoPanel::draw(const PanelDrawContext& ctx) {
        if (ctx.ui && ctx.viewport)
            gizmo_->renderNodeTransformGizmo(*ctx.ui, *ctx.viewport);
    }

    CropBoxGizmoPanel::CropBoxGizmoPanel(GizmoManager* gizmo)
        : gizmo_(gizmo) {}

    void CropBoxGizmoPanel::draw(const PanelDrawContext& ctx) {
        if (ctx.ui && ctx.viewport)
            gizmo_->renderCropBoxGizmo(*ctx.ui, *ctx.viewport);
    }

    EllipsoidGizmoPanel::EllipsoidGizmoPanel(GizmoManager* gizmo)
        : gizmo_(gizmo) {}

    void EllipsoidGizmoPanel::draw(const PanelDrawContext& ctx) {
        if (ctx.ui && ctx.viewport)
            gizmo_->renderEllipsoidGizmo(*ctx.ui, *ctx.viewport);
    }

    ViewportGizmoPanel::ViewportGizmoPanel(GizmoManager* gizmo)
        : gizmo_(gizmo) {}

    void ViewportGizmoPanel::draw(const PanelDrawContext& ctx) {
        if (ctx.viewport)
            gizmo_->renderViewportGizmo(*ctx.viewport);
    }

    bool ViewportGizmoPanel::poll(const PanelDrawContext& ctx) {
        return !ctx.ui_hidden && ctx.viewport &&
               ctx.viewport->size.x > 0 && ctx.viewport->size.y > 0;
    }

    PieMenuPanel::PieMenuPanel(GizmoManager* gizmo)
        : gizmo_(gizmo) {}

    void PieMenuPanel::draw(const PanelDrawContext&) {
        gizmo_->renderPieMenu();
    }

    bool PieMenuPanel::poll(const PanelDrawContext&) {
        return gizmo_->isPieMenuOpen();
    }

    PythonOverlayPanel::PythonOverlayPanel(GuiManager* gui)
        : gui_(gui) {}

    bool PythonOverlayPanel::poll(const PanelDrawContext& ctx) {
        if (gui_ && gui_->isStartupVisible()) {
            return false;
        }
        return ctx.viewport && ctx.viewport->size.x > 0 && ctx.viewport->size.y > 0 &&
               python::has_viewport_draw_handlers();
    }

    void PythonOverlayPanel::draw(const PanelDrawContext& ctx) {
        if (!ctx.ui || !ctx.ui->viewer || !ctx.viewport)
            return;

        const auto& vp = ctx.ui->viewer->getViewport();
        const auto view = vp.getViewMatrix();
        auto* rm = ctx.ui->viewer->getRenderingManager();
        const float focal_mm = rm ? rm->getFocalLengthMm() : lfs::rendering::DEFAULT_FOCAL_LENGTH_MM;
        const auto proj = vp.getProjectionMatrix(focal_mm);
        const float vp_pos[] = {ctx.viewport->pos.x, ctx.viewport->pos.y};
        const float vp_size[] = {ctx.viewport->size.x, ctx.viewport->size.y};
        const float cam_pos[] = {vp.camera.t.x, vp.camera.t.y, vp.camera.t.z};
        const glm::vec3 forward = lfs::rendering::cameraForward(vp.camera.R);
        const float cam_fwd[] = {forward.x, forward.y, forward.z};

        python::invoke_viewport_overlay(glm::value_ptr(view), glm::value_ptr(proj),
                                        vp_pos, vp_size, cam_pos, cam_fwd,
                                        ImGui::GetBackgroundDrawList());
    }

} // namespace lfs::vis::gui::native_panels
