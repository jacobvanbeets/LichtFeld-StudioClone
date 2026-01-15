/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "sequencer_command.hpp"
#include "core/services.hpp"
#include "gui/gui_manager.hpp"

namespace lfs::vis::command {

    namespace {
        constexpr float TIME_EPSILON = 0.001f;

        [[nodiscard]] sequencer::Timeline* getTimeline() {
            if (auto* const gm = services().guiOrNull()) {
                return &gm->sequencer().timeline();
            }
            return nullptr;
        }
    } // namespace

    // AddKeyframeCommand

    AddKeyframeCommand::AddKeyframeCommand(sequencer::Keyframe keyframe)
        : keyframe_(std::move(keyframe)) {}

    void AddKeyframeCommand::undo() {
        if (auto* tl = getTimeline()) {
            // Find and remove the keyframe with matching time
            const auto& kfs = tl->keyframes();
            for (size_t i = 0; i < kfs.size(); ++i) {
                if (std::abs(kfs[i].time - keyframe_.time) < TIME_EPSILON) {
                    tl->removeKeyframe(i);
                    break;
                }
            }
        }
    }

    void AddKeyframeCommand::redo() {
        if (auto* tl = getTimeline()) {
            tl->addKeyframe(keyframe_);
        }
    }

    // RemoveKeyframeCommand

    RemoveKeyframeCommand::RemoveKeyframeCommand(size_t index, sequencer::Keyframe keyframe)
        : index_(index),
          keyframe_(std::move(keyframe)) {}

    void RemoveKeyframeCommand::undo() {
        if (auto* tl = getTimeline()) {
            tl->addKeyframe(keyframe_);
        }
    }

    void RemoveKeyframeCommand::redo() {
        if (auto* tl = getTimeline()) {
            // Find keyframe by time since index may have changed
            const auto& kfs = tl->keyframes();
            for (size_t i = 0; i < kfs.size(); ++i) {
                if (std::abs(kfs[i].time - keyframe_.time) < TIME_EPSILON) {
                    tl->removeKeyframe(i);
                    break;
                }
            }
        }
    }

    // UpdateKeyframeCommand

    UpdateKeyframeCommand::UpdateKeyframeCommand(size_t index,
                                                 sequencer::Keyframe old_keyframe,
                                                 sequencer::Keyframe new_keyframe)
        : index_(index),
          old_keyframe_(std::move(old_keyframe)),
          new_keyframe_(std::move(new_keyframe)) {}

    void UpdateKeyframeCommand::undo() {
        if (auto* tl = getTimeline()) {
            const auto& kfs = tl->keyframes();
            for (size_t i = 0; i < kfs.size(); ++i) {
                if (std::abs(kfs[i].time - new_keyframe_.time) < TIME_EPSILON) {
                    tl->updateKeyframe(i, old_keyframe_.position, old_keyframe_.rotation, old_keyframe_.fov);
                    break;
                }
            }
        }
    }

    void UpdateKeyframeCommand::redo() {
        if (auto* tl = getTimeline()) {
            const auto& kfs = tl->keyframes();
            for (size_t i = 0; i < kfs.size(); ++i) {
                if (std::abs(kfs[i].time - old_keyframe_.time) < TIME_EPSILON) {
                    tl->updateKeyframe(i, new_keyframe_.position, new_keyframe_.rotation, new_keyframe_.fov);
                    break;
                }
            }
        }
    }

    // MoveKeyframeCommand

    MoveKeyframeCommand::MoveKeyframeCommand(size_t old_index, float old_time, float new_time)
        : old_index_(old_index),
          old_time_(old_time),
          new_time_(new_time) {}

    void MoveKeyframeCommand::undo() {
        if (auto* tl = getTimeline()) {
            const auto& kfs = tl->keyframes();
            for (size_t i = 0; i < kfs.size(); ++i) {
                if (std::abs(kfs[i].time - new_time_) < TIME_EPSILON) {
                    tl->setKeyframeTime(i, old_time_);
                    break;
                }
            }
        }
    }

    void MoveKeyframeCommand::redo() {
        if (auto* tl = getTimeline()) {
            const auto& kfs = tl->keyframes();
            for (size_t i = 0; i < kfs.size(); ++i) {
                if (std::abs(kfs[i].time - old_time_) < TIME_EPSILON) {
                    tl->setKeyframeTime(i, new_time_);
                    break;
                }
            }
        }
    }

    // SetKeyframeEasingCommand

    SetKeyframeEasingCommand::SetKeyframeEasingCommand(size_t index,
                                                       sequencer::EasingType old_easing,
                                                       sequencer::EasingType new_easing)
        : index_(index),
          old_easing_(old_easing),
          new_easing_(new_easing) {}

    void SetKeyframeEasingCommand::undo() {
        if (auto* tl = getTimeline()) {
            if (index_ < tl->size()) {
                tl->setKeyframeEasing(index_, old_easing_);
            }
        }
    }

    void SetKeyframeEasingCommand::redo() {
        if (auto* tl = getTimeline()) {
            if (index_ < tl->size()) {
                tl->setKeyframeEasing(index_, new_easing_);
            }
        }
    }

} // namespace lfs::vis::command
