#version 450

// Fragment stage for GPU-instanced camera frustums. Mirrors shape_overlay.frag:
// signed-distance anti-aliased line segments that fade where they pass behind
// dense splats. The only difference is the push-constant layout, which carries
// the view-projection matrix used by frustum.vert.

layout(location = 0) in vec2 ScreenPos;
layout(location = 1) in vec2 P0;
layout(location = 2) in vec2 P1;
layout(location = 3) in vec4 Color;
layout(location = 4) in vec4 Params;
layout(location = 5) in float ViewDepth;
layout(location = 0) out vec4 FragColor;

layout(set = 1, binding = 0) uniform sampler2D u_splat_depth;

layout(push_constant) uniform FrustumPush {
    mat4 view;
    vec4 viewport_panel;
    vec4 viewport_fb;
    vec4 projection;
    // x: depth available, y: flip-y depth UV, z: line thickness, w: orthographic.
    vec4 params;
} pc;

float sdSegment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h);
}

void main() {
    float thickness = max(Params.y, 1.0);
    float aa = max(Params.w, 0.75);
    float signed_dist = sdSegment(ScreenPos, P0, P1) - thickness * 0.5;
    float alpha = Color.a * smoothstep(aa, -aa, signed_dist);

    // Soft passthrough: frustum lines in front of splats render at full opacity,
    // lines behind dense splats fade so the wireframe stays readable through the
    // scene. ViewDepth = 0 (orthographic / no depth) skips the comparison.
    if (pc.params.x > 0.5 && ViewDepth > 0.0) {
        vec2 vp = max(pc.viewport_fb.zw, vec2(1.0));
        vec2 uv = (gl_FragCoord.xy - pc.viewport_fb.xy) / vp;
        if (pc.params.y > 0.5) {
            uv.y = 1.0 - uv.y;
        }
        float splat_depth = texture(u_splat_depth, uv).r;
        if (splat_depth > 0.0 && splat_depth < 1.0e9) {
            float fade = smoothstep(0.01, 0.15, ViewDepth - splat_depth);
            alpha *= mix(1.0, 0.25, fade);
        }
    }

    if (alpha <= 0.001) {
        discard;
    }
    FragColor = vec4(Color.rgb, alpha);
}
