/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "pivot_renderer.hpp"
#include "core/logger.hpp"
#include "gl_state_guard.hpp"
#include <glad/glad.h>

namespace lfs::rendering {

    namespace {

        // Billboard quad with fixed screen size
        constexpr const char* VERT_SRC = R"(
#version 330 core

uniform mat4 u_view;
uniform mat4 u_projection;
uniform vec3 u_pivot_pos;
uniform float u_screen_size;
uniform vec2 u_viewport_size;

out vec2 v_uv;

const vec2 CORNERS[4] = vec2[4](
    vec2(-1.0, -1.0), vec2(1.0, -1.0),
    vec2(1.0, 1.0), vec2(-1.0, 1.0)
);
const int INDICES[6] = int[6](0, 1, 2, 0, 2, 3);

void main() {
    vec2 corner = CORNERS[INDICES[gl_VertexID]];
    v_uv = corner;

    vec4 clip_pos = u_projection * u_view * vec4(u_pivot_pos, 1.0);
    vec2 ndc = clip_pos.xy / clip_pos.w;
    vec2 size_ndc = (u_screen_size / u_viewport_size) * 2.0;

    gl_Position = vec4(ndc + corner * size_ndc, 0.0, 1.0);
}
)";

        // Ripple effect with flash and expanding rings
        constexpr const char* FRAG_SRC = R"(
#version 330 core

uniform vec3 u_color;
uniform float u_opacity;

in vec2 v_uv;
out vec4 FragColor;

const float DOT_RADIUS = 0.06;
const float RING_WIDTH = 0.045;
const float MAX_RADIUS = 0.92;
const float EDGE_AA = 0.015;

float ring(float dist, float radius, float width) {
    float inner = radius - width * 0.5;
    float outer = radius + width * 0.5;
    return smoothstep(inner - EDGE_AA, inner + EDGE_AA, dist) *
           (1.0 - smoothstep(outer - EDGE_AA, outer + EDGE_AA, dist));
}

void main() {
    float dist = length(v_uv);
    float progress = 1.0 - u_opacity;

    // Flash burst
    float flash_intensity = pow(max(0.0, 1.0 - progress * 4.0), 2.0);
    float flash = (1.0 - smoothstep(0.0, 0.25, dist)) * flash_intensity;

    // Center dot with pulse
    float dot_scale = 1.0 + 0.3 * sin(progress * 6.28) * (1.0 - progress);
    float dot_radius = DOT_RADIUS * dot_scale;
    float dot_alpha = (1.0 - smoothstep(dot_radius - EDGE_AA, dot_radius + EDGE_AA, dist));
    dot_alpha *= pow(u_opacity, 0.5);

    // Expanding rings
    float r1_prog = clamp(progress * 1.5, 0.0, 1.0);
    float r1_radius = DOT_RADIUS + r1_prog * (MAX_RADIUS - DOT_RADIUS);
    float r1_alpha = ring(dist, r1_radius, RING_WIDTH) * pow(1.0 - r1_prog, 1.5);

    float r2_prog = clamp((progress - 0.15) * 1.5, 0.0, 1.0);
    float r2_radius = DOT_RADIUS + r2_prog * (MAX_RADIUS * 0.85 - DOT_RADIUS);
    float r2_alpha = ring(dist, r2_radius, RING_WIDTH * 0.7) * pow(1.0 - r2_prog, 1.5) * 0.6;

    // Glow
    float glow = exp(-pow((dist - r1_radius * 0.95) * 4.0, 2.0)) * (1.0 - r1_prog) * 0.3;

    float alpha = max(max(dot_alpha, flash), max(r1_alpha + r2_alpha, glow));
    if (alpha < 0.01) discard;

    vec3 color = u_color + vec3(0.4) * (flash + dot_alpha * 0.3);
    FragColor = vec4(color, alpha);
}
)";

        GLuint compileShader(const GLenum type, const char* source) {
            const GLuint shader = glCreateShader(type);
            glShaderSource(shader, 1, &source, nullptr);
            glCompileShader(shader);

            GLint success;
            glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
            if (!success) {
                char log[512];
                glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
                LOG_ERROR("Shader compile error: {}", log);
                glDeleteShader(shader);
                return 0;
            }
            return shader;
        }

    } // namespace

    Result<void> RenderPivotPoint::init() {
        if (initialized_) {
            return {};
        }

        const GLuint vert = compileShader(GL_VERTEX_SHADER, VERT_SRC);
        if (vert == 0) {
            return std::unexpected("Failed to compile pivot vertex shader");
        }

        const GLuint frag = compileShader(GL_FRAGMENT_SHADER, FRAG_SRC);
        if (frag == 0) {
            glDeleteShader(vert);
            return std::unexpected("Failed to compile pivot fragment shader");
        }

        shader_program_ = glCreateProgram();
        glAttachShader(shader_program_, vert);
        glAttachShader(shader_program_, frag);
        glLinkProgram(shader_program_);
        glDeleteShader(vert);
        glDeleteShader(frag);

        GLint success;
        glGetProgramiv(shader_program_, GL_LINK_STATUS, &success);
        if (!success) {
            char log[512];
            glGetProgramInfoLog(shader_program_, sizeof(log), nullptr, log);
            LOG_ERROR("Shader link error: {}", log);
            glDeleteProgram(shader_program_);
            shader_program_ = 0;
            return std::unexpected("Failed to link pivot shader");
        }

        // Cache uniform locations
        loc_view_ = glGetUniformLocation(shader_program_, "u_view");
        loc_projection_ = glGetUniformLocation(shader_program_, "u_projection");
        loc_pivot_pos_ = glGetUniformLocation(shader_program_, "u_pivot_pos");
        loc_screen_size_ = glGetUniformLocation(shader_program_, "u_screen_size");
        loc_viewport_size_ = glGetUniformLocation(shader_program_, "u_viewport_size");
        loc_color_ = glGetUniformLocation(shader_program_, "u_color");
        loc_opacity_ = glGetUniformLocation(shader_program_, "u_opacity");

        auto vao_result = create_vao();
        if (!vao_result) {
            return std::unexpected(vao_result.error());
        }
        vao_ = std::move(*vao_result);

        initialized_ = true;
        LOG_DEBUG("Pivot renderer initialized");
        return {};
    }

    Result<void> RenderPivotPoint::render(const glm::mat4& view, const glm::mat4& projection) {
        if (!initialized_) {
            return std::unexpected("Pivot renderer not initialized");
        }

        GLint viewport[4];
        glGetIntegerv(GL_VIEWPORT, viewport);
        const glm::vec2 viewport_size(static_cast<float>(viewport[2]), static_cast<float>(viewport[3]));

        // Save GL state
        const GLboolean depth_enabled = glIsEnabled(GL_DEPTH_TEST);
        const GLboolean blend_enabled = glIsEnabled(GL_BLEND);
        GLint blend_src, blend_dst;
        glGetIntegerv(GL_BLEND_SRC_ALPHA, &blend_src);
        glGetIntegerv(GL_BLEND_DST_ALPHA, &blend_dst);

        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glUseProgram(shader_program_);
        glUniformMatrix4fv(loc_view_, 1, GL_FALSE, &view[0][0]);
        glUniformMatrix4fv(loc_projection_, 1, GL_FALSE, &projection[0][0]);
        glUniform3fv(loc_pivot_pos_, 1, &pivot_position_[0]);
        glUniform1f(loc_screen_size_, screen_size_);
        glUniform2fv(loc_viewport_size_, 1, &viewport_size[0]);
        glUniform3fv(loc_color_, 1, &color_[0]);
        glUniform1f(loc_opacity_, opacity_);

        VAOBinder vao_bind(vao_);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        // Restore GL state
        if (depth_enabled)
            glEnable(GL_DEPTH_TEST);
        if (!blend_enabled)
            glDisable(GL_BLEND);
        glBlendFunc(blend_src, blend_dst);

        return {};
    }

} // namespace lfs::rendering
