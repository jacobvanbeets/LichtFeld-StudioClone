/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#ifdef LFS_VR_ENABLED

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <string>

// Forward declare OpenVR types to avoid header pollution
namespace vr {
    class IVRSystem;
    class IVRCompositor;
    struct TrackedDevicePose_t;
} // namespace vr

namespace lfs::vis::vr {

    enum class VREye { Left = 0, Right = 1 };

    /// Per-controller state snapshot
    struct VRControllerState {
        bool valid = false;
        glm::mat4 transform{1.0f};

        // Buttons (Quest 3 via Virtual Desktop / SteamVR)
        bool trigger_pressed = false;
        bool grip_pressed = false;
        float trigger_value = 0.0f;
        float grip_value = 0.0f;
        glm::vec2 thumbstick{0.0f, 0.0f};
        bool thumbstick_pressed = false;
        bool button_a = false;
        bool button_b = false;
        bool menu_button = false;
    };

    /// Manages the OpenVR session, HMD tracking, stereo FBOs, and frame submission.
    /// Adapted from Splatshop's OpenVRHelper for LichtFeld conventions.
    class VRManager {
    public:
        VRManager();
        ~VRManager();

        // Non-copyable
        VRManager(const VRManager&) = delete;
        VRManager& operator=(const VRManager&) = delete;

        // --- Lifecycle ---
        /// Initialize OpenVR and create stereo framebuffers. Returns false on failure.
        bool start();
        /// Shutdown OpenVR and release GPU resources.
        void stop();
        /// True if VR session is active and HMD is connected.
        [[nodiscard]] bool is_active() const;

        // --- HMD queries ---
        /// Recommended per-eye render resolution from the HMD.
        [[nodiscard]] glm::ivec2 get_render_size() const { return render_size_; }
        /// Projection matrix for given eye (uses HMD lens parameters).
        [[nodiscard]] glm::mat4 get_eye_projection(VREye eye, float near_plane = 0.05f, float far_plane = 500.0f) const;
        /// Eye-to-head offset transform (IPD-based stereo separation).
        [[nodiscard]] glm::mat4 get_eye_to_head(VREye eye) const;
        /// Current HMD world pose (head position + orientation). Updated by process_events().
        [[nodiscard]] glm::mat4 get_hmd_pose() const { return hmd_pose_; }

        // --- Frame lifecycle ---
        /// Poll VR events and update device poses. Call once per frame before rendering.
        void process_events();
        /// Submit rendered left/right eye OpenGL texture IDs to the VR compositor.
        void submit_frames(unsigned int left_texture_id, unsigned int right_texture_id);

        // --- Stereo FBO management ---
        /// Get the FBO handle for a given eye. Render into this before calling submit.
        [[nodiscard]] unsigned int get_eye_fbo(VREye eye) const;
        /// Get the color texture from an eye FBO (for desktop mirror / submission).
        [[nodiscard]] unsigned int get_eye_texture(VREye eye) const;
        /// Bind the FBO for a given eye and set the viewport.
        void bind_eye_fbo(VREye eye);

        // --- Controllers ---
        [[nodiscard]] const VRControllerState& get_left_controller() const { return left_controller_; }
        [[nodiscard]] const VRControllerState& get_right_controller() const { return right_controller_; }

        /// Last error message (for UI display).
        [[nodiscard]] const std::string& get_last_error() const { return last_error_; }

    private:
        void create_stereo_fbos();
        void destroy_stereo_fbos();
        void update_controller_state(int device_index, VRControllerState& state);
        glm::mat4 openvr_to_glm(const float (&mat)[3][4]) const;

        ::vr::IVRSystem* system_ = nullptr;
        ::vr::IVRCompositor* compositor_ = nullptr;

        // HMD state
        glm::mat4 hmd_pose_{1.0f};
        glm::ivec2 render_size_{0, 0};

        // Stereo framebuffers [0]=left, [1]=right
        unsigned int eye_fbo_[2] = {0, 0};
        unsigned int eye_color_tex_[2] = {0, 0};
        unsigned int eye_depth_tex_[2] = {0, 0};

        // Controller state
        VRControllerState left_controller_;
        VRControllerState right_controller_;

        std::string last_error_;
    };

} // namespace lfs::vis::vr

#endif // LFS_VR_ENABLED
