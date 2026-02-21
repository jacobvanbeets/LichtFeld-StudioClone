/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#ifdef LFS_VR_ENABLED

#include "vr/vr_manager.hpp"
#include "core/logger.hpp"

#include <openvr.h>

#include <glad/glad.h>
#include <glm/gtc/type_ptr.hpp>

namespace lfs::vis::vr {

    VRManager::VRManager() = default;

    VRManager::~VRManager() {
        stop();
    }

    bool VRManager::start() {
        if (is_active()) {
            LOG_WARN("VR session already active");
            return true;
        }

        // Initialize OpenVR
        ::vr::EVRInitError error;
        system_ = ::vr::VR_Init(&error, ::vr::VRApplication_Scene);

        if (error != ::vr::VRInitError_None) {
            last_error_ = ::vr::VR_GetVRInitErrorAsEnglishDescription(error);
            LOG_ERROR("Failed to initialize OpenVR: {}", last_error_);
            system_ = nullptr;
            return false;
        }

        // Get compositor
        compositor_ = ::vr::VRCompositor();
        if (!compositor_) {
            last_error_ = "Failed to initialize VR compositor";
            LOG_ERROR("{}", last_error_);
            ::vr::VR_Shutdown();
            system_ = nullptr;
            return false;
        }

        // Query recommended render target size per eye
        uint32_t w = 0, h = 0;
        system_->GetRecommendedRenderTargetSize(&w, &h);
        render_size_ = glm::ivec2(static_cast<int>(w), static_cast<int>(h));
        LOG_INFO("VR initialized - per-eye resolution: {}x{}", render_size_.x, render_size_.y);

        // Log HMD info
        char buf[256];
        auto get_string = [&](::vr::ETrackedDeviceProperty prop) -> std::string {
            ::vr::ETrackedPropertyError err;
            system_->GetStringTrackedDeviceProperty(::vr::k_unTrackedDeviceIndex_Hmd, prop, buf, sizeof(buf), &err);
            return (err == ::vr::TrackedProp_Success) ? std::string(buf) : std::string("unknown");
        };
        LOG_INFO("HMD: {} - {}", get_string(::vr::Prop_TrackingSystemName_String),
                 get_string(::vr::Prop_SerialNumber_String));

        // Create stereo framebuffers
        create_stereo_fbos();

        return true;
    }

    void VRManager::stop() {
        destroy_stereo_fbos();

        if (system_) {
            ::vr::VR_Shutdown();
            system_ = nullptr;
            compositor_ = nullptr;
            LOG_INFO("VR session stopped");
        }
    }

    bool VRManager::is_active() const {
        return system_ != nullptr;
    }

    // --- Stereo FBO management ---

    void VRManager::create_stereo_fbos() {
        if (render_size_.x <= 0 || render_size_.y <= 0)
            return;

        for (int i = 0; i < 2; ++i) {
            // Color texture
            glGenTextures(1, &eye_color_tex_[i]);
            glBindTexture(GL_TEXTURE_2D, eye_color_tex_[i]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, render_size_.x, render_size_.y, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

            // Depth texture
            glGenTextures(1, &eye_depth_tex_[i]);
            glBindTexture(GL_TEXTURE_2D, eye_depth_tex_[i]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, render_size_.x, render_size_.y, 0,
                         GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

            // FBO
            glGenFramebuffers(1, &eye_fbo_[i]);
            glBindFramebuffer(GL_FRAMEBUFFER, eye_fbo_[i]);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, eye_color_tex_[i], 0);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, eye_depth_tex_[i], 0);

            GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
            if (status != GL_FRAMEBUFFER_COMPLETE) {
                LOG_ERROR("VR FBO {} incomplete: 0x{:X}", i, status);
            }
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glBindTexture(GL_TEXTURE_2D, 0);
        LOG_DEBUG("VR stereo FBOs created ({}x{})", render_size_.x, render_size_.y);
    }

    void VRManager::destroy_stereo_fbos() {
        for (int i = 0; i < 2; ++i) {
            if (eye_fbo_[i]) {
                glDeleteFramebuffers(1, &eye_fbo_[i]);
                eye_fbo_[i] = 0;
            }
            if (eye_color_tex_[i]) {
                glDeleteTextures(1, &eye_color_tex_[i]);
                eye_color_tex_[i] = 0;
            }
            if (eye_depth_tex_[i]) {
                glDeleteTextures(1, &eye_depth_tex_[i]);
                eye_depth_tex_[i] = 0;
            }
        }
    }

    unsigned int VRManager::get_eye_fbo(VREye eye) const {
        return eye_fbo_[static_cast<int>(eye)];
    }

    unsigned int VRManager::get_eye_texture(VREye eye) const {
        return eye_color_tex_[static_cast<int>(eye)];
    }

    void VRManager::bind_eye_fbo(VREye eye) {
        glBindFramebuffer(GL_FRAMEBUFFER, get_eye_fbo(eye));
        glViewport(0, 0, render_size_.x, render_size_.y);
    }

    // --- HMD pose and projection ---

    glm::mat4 VRManager::openvr_to_glm(const float (&m)[3][4]) const {
        // OpenVR uses row-major 3x4 matrices; convert to GLM column-major 4x4
        return glm::mat4(
            m[0][0], m[1][0], m[2][0], 0.0f,
            m[0][1], m[1][1], m[2][1], 0.0f,
            m[0][2], m[1][2], m[2][2], 0.0f,
            m[0][3], m[1][3], m[2][3], 1.0f);
    }

    glm::mat4 VRManager::get_eye_projection(VREye eye, float near_plane, float far_plane) const {
        if (!system_)
            return glm::mat4(1.0f);

        ::vr::HmdMatrix44_t mat = system_->GetProjectionMatrix(
            static_cast<::vr::EVREye>(eye), near_plane, far_plane);

        // OpenVR projection is row-major; transpose to column-major for GLM
        return glm::mat4(
            mat.m[0][0], mat.m[1][0], mat.m[2][0], mat.m[3][0],
            mat.m[0][1], mat.m[1][1], mat.m[2][1], mat.m[3][1],
            mat.m[0][2], mat.m[1][2], mat.m[2][2], mat.m[3][2],
            mat.m[0][3], mat.m[1][3], mat.m[2][3], mat.m[3][3]);
    }

    glm::mat4 VRManager::get_eye_to_head(VREye eye) const {
        if (!system_)
            return glm::mat4(1.0f);

        ::vr::HmdMatrix34_t mat = system_->GetEyeToHeadTransform(
            static_cast<::vr::EVREye>(eye));
        return openvr_to_glm(mat.m);
    }

    // --- Event processing and pose updates ---

    void VRManager::process_events() {
        if (!system_)
            return;

        // Poll VR events
        ::vr::VREvent_t event;
        while (system_->PollNextEvent(&event, sizeof(event))) {
            // Could handle specific events (e.g., TrackedDeviceActivated) here
        }

        // Update device poses - WaitGetPoses blocks until the next VSync
        ::vr::TrackedDevicePose_t poses[::vr::k_unMaxTrackedDeviceCount];
        compositor_->WaitGetPoses(poses, ::vr::k_unMaxTrackedDeviceCount, nullptr, 0);

        // Extract HMD pose (device index 0)
        if (poses[::vr::k_unTrackedDeviceIndex_Hmd].bPoseIsValid) {
            hmd_pose_ = openvr_to_glm(poses[::vr::k_unTrackedDeviceIndex_Hmd].mDeviceToAbsoluteTracking.m);
        }

        // Find and update controllers
        int left_id = system_->GetTrackedDeviceIndexForControllerRole(
            ::vr::TrackedControllerRole_LeftHand);
        int right_id = system_->GetTrackedDeviceIndexForControllerRole(
            ::vr::TrackedControllerRole_RightHand);

        auto update_controller = [&](int device_id, VRControllerState& state) {
            if (device_id < 0 || device_id >= static_cast<int>(::vr::k_unMaxTrackedDeviceCount)) {
                state.valid = false;
                return;
            }

            if (!poses[device_id].bPoseIsValid) {
                state.valid = false;
                return;
            }

            state.valid = true;
            state.transform = openvr_to_glm(poses[device_id].mDeviceToAbsoluteTracking.m);

            // Read button/axis state via legacy API (works with Virtual Desktop / VDXR)
            ::vr::VRControllerState_t ctrl_state;
            if (system_->GetControllerState(device_id, &ctrl_state, sizeof(ctrl_state))) {
                // Trigger: axis 1
                state.trigger_value = ctrl_state.rAxis[1].x;
                state.trigger_pressed = (ctrl_state.ulButtonPressed &
                                         ::vr::ButtonMaskFromId(::vr::k_EButton_SteamVR_Trigger)) != 0;

                // Grip: button mask
                state.grip_pressed = (ctrl_state.ulButtonPressed &
                                      ::vr::ButtonMaskFromId(::vr::k_EButton_Grip)) != 0;
                state.grip_value = state.grip_pressed ? 1.0f : 0.0f;

                // Thumbstick: axis 0
                state.thumbstick = glm::vec2(ctrl_state.rAxis[0].x, ctrl_state.rAxis[0].y);
                state.thumbstick_pressed = (ctrl_state.ulButtonPressed &
                                            ::vr::ButtonMaskFromId(::vr::k_EButton_SteamVR_Touchpad)) != 0;

                // A/B buttons (ApplicationMenu = B, Dashboard = A on some mappings)
                state.button_a = (ctrl_state.ulButtonPressed &
                                  ::vr::ButtonMaskFromId(::vr::k_EButton_A)) != 0;
                state.menu_button = (ctrl_state.ulButtonPressed &
                                     ::vr::ButtonMaskFromId(::vr::k_EButton_ApplicationMenu)) != 0;
            }
        };

        update_controller(left_id, left_controller_);
        update_controller(right_id, right_controller_);
    }

    // --- Frame submission ---

    void VRManager::submit_frames(unsigned int left_texture_id, unsigned int right_texture_id) {
        if (!compositor_)
            return;

        ::vr::Texture_t left_tex = {
            reinterpret_cast<void*>(static_cast<uintptr_t>(left_texture_id)),
            ::vr::TextureType_OpenGL,
            ::vr::ColorSpace_Gamma};

        ::vr::Texture_t right_tex = {
            reinterpret_cast<void*>(static_cast<uintptr_t>(right_texture_id)),
            ::vr::TextureType_OpenGL,
            ::vr::ColorSpace_Gamma};

        compositor_->Submit(::vr::Eye_Left, &left_tex);
        compositor_->Submit(::vr::Eye_Right, &right_tex);
    }

} // namespace lfs::vis::vr

#endif // LFS_VR_ENABLED
