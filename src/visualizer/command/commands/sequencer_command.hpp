/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "command/command.hpp"
#include "sequencer/keyframe.hpp"
#include <vector>

namespace lfs::vis::command {

    class AddKeyframeCommand : public Command {
    public:
        AddKeyframeCommand(sequencer::Keyframe keyframe);

        void undo() override;
        void redo() override;
        std::string getName() const override { return "Add Keyframe"; }

    private:
        sequencer::Keyframe keyframe_;
        size_t inserted_index_ = 0;
    };

    class RemoveKeyframeCommand : public Command {
    public:
        RemoveKeyframeCommand(size_t index, sequencer::Keyframe keyframe);

        void undo() override;
        void redo() override;
        std::string getName() const override { return "Remove Keyframe"; }

    private:
        size_t index_;
        sequencer::Keyframe keyframe_;
    };

    class UpdateKeyframeCommand : public Command {
    public:
        UpdateKeyframeCommand(size_t index,
                              sequencer::Keyframe old_keyframe,
                              sequencer::Keyframe new_keyframe);

        void undo() override;
        void redo() override;
        std::string getName() const override { return "Update Keyframe"; }

    private:
        size_t index_;
        sequencer::Keyframe old_keyframe_;
        sequencer::Keyframe new_keyframe_;
    };

    class MoveKeyframeCommand : public Command {
    public:
        MoveKeyframeCommand(size_t old_index, float old_time, float new_time);

        void undo() override;
        void redo() override;
        std::string getName() const override { return "Move Keyframe"; }

    private:
        size_t old_index_;
        float old_time_;
        float new_time_;
    };

    class SetKeyframeEasingCommand : public Command {
    public:
        SetKeyframeEasingCommand(size_t index,
                                 sequencer::EasingType old_easing,
                                 sequencer::EasingType new_easing);

        void undo() override;
        void redo() override;
        std::string getName() const override { return "Set Easing"; }

    private:
        size_t index_;
        sequencer::EasingType old_easing_;
        sequencer::EasingType new_easing_;
    };

} // namespace lfs::vis::command
