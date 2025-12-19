/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "keyframe.hpp"
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace lfs::sequencer {

    inline constexpr int DEFAULT_PATH_SAMPLES = 20;

    class Timeline {
    public:
        void addKeyframe(const Keyframe& keyframe);
        void removeKeyframe(size_t index);
        void setKeyframeTime(size_t index, float new_time, bool sort = true);
        void updateKeyframe(size_t index, const glm::vec3& position, const glm::quat& rotation, float fov);
        void sortKeyframes();
        void clear();

        [[nodiscard]] bool empty() const { return keyframes_.empty(); }
        [[nodiscard]] size_t size() const { return keyframes_.size(); }
        [[nodiscard]] std::span<const Keyframe> keyframes() const { return keyframes_; }

        [[nodiscard]] float duration() const;
        [[nodiscard]] float startTime() const;
        [[nodiscard]] float endTime() const;

        [[nodiscard]] CameraState evaluate(float time) const;
        [[nodiscard]] std::vector<glm::vec3> generatePath(int samples_per_segment = DEFAULT_PATH_SAMPLES) const;

        [[nodiscard]] bool saveToJson(const std::string& path) const;
        [[nodiscard]] bool loadFromJson(const std::string& path);

    private:
        std::vector<Keyframe> keyframes_;
    };

} // namespace lfs::sequencer
