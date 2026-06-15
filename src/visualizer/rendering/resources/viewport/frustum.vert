#version 450

// GPU-instanced camera frustum wireframes. Each instance is one camera; this
// shader generates the 8 frustum edges (48 vertices = 8 edges * 2 triangles)
// from gl_VertexIndex, projects the edge endpoints with the same camera-space
// pinhole math as the former CPU overlay path, clips them to the near plane,
// expands each edge into a thick
// screen-space quad, and emits the same varyings the SDF line fragment shader
// consumes. This replaces the former per-frame CPU projection + tessellation of
// every camera frustum (~160k vertices/frame for large datasets).

layout(location = 0) out vec2 ScreenPos;
layout(location = 1) out vec2 P0;
layout(location = 2) out vec2 P1;
layout(location = 3) out vec4 Color;
layout(location = 4) out vec4 Params;
layout(location = 5) out float ViewDepth;

struct FrustumInstance {
    mat4 model;
    vec4 color;
};

layout(std430, set = 0, binding = 0) readonly buffer FrustumInstances {
    FrustumInstance items[];
} frustums;

layout(push_constant) uniform FrustumPush {
    mat4 view;
    // x,y: panel origin (logical px). z,w: panel size (logical px).
    vec4 viewport_panel;
    // x,y: viewport origin (framebuffer px). z,w: viewport size (framebuffer px).
    vec4 viewport_fb;
    // x,y: render size (px). z,w: perspective fx/fy, or z = ortho scale when orthographic.
    vec4 projection;
    // x: depth available, y: flip-y depth UV, z: line thickness (px), w: orthographic.
    vec4 params;
} pc;

// 5 local frustum points: 4 image-plane corners (z = -1) + apex at the origin.
const vec3 PTS[5] = vec3[5](
    vec3(-0.5, -0.5, -1.0),
    vec3( 0.5, -0.5, -1.0),
    vec3( 0.5,  0.5, -1.0),
    vec3(-0.5,  0.5, -1.0),
    vec3( 0.0,  0.0,  0.0));

// 8 edges: image-plane rectangle (4) + apex-to-corner (4).
const ivec2 EDGES[8] = ivec2[8](
    ivec2(0, 1), ivec2(1, 2), ivec2(2, 3), ivec2(3, 0),
    ivec2(0, 4), ivec2(1, 4), ivec2(2, 4), ivec2(3, 4));

// 6 quad corners per edge as (end, side): end 0 = P0, end 1 = P1; side +/-1.
const vec2 QUAD[6] = vec2[6](
    vec2(0.0,  1.0), vec2(1.0,  1.0), vec2(1.0, -1.0),
    vec2(0.0,  1.0), vec2(1.0, -1.0), vec2(0.0, -1.0));

void emitDegenerate() {
    ScreenPos = vec2(0.0);
    P0 = vec2(0.0);
    P1 = vec2(0.0);
    Color = vec4(0.0);
    Params = vec4(0.0, 1.0, 0.0, 1.0);
    ViewDepth = 0.0;
    // z outside [0,1] -> the whole primitive is clipped away.
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
}

void main() {
    uint edge = uint(gl_VertexIndex) / 6u;
    uint corner = uint(gl_VertexIndex) % 6u;
    FrustumInstance inst = frustums.items[gl_InstanceIndex];

    ivec2 e = EDGES[edge];
    vec4 world_a = inst.model * vec4(PTS[e.x], 1.0);
    vec4 world_b = inst.model * vec4(PTS[e.y], 1.0);
    vec3 view_a = (pc.view * world_a).xyz;
    vec3 view_b = (pc.view * world_b).xyz;

    const float kMinViewZ = -1e-4;
    bool a_behind = view_a.z >= kMinViewZ;
    bool b_behind = view_b.z >= kMinViewZ;
    if (a_behind && b_behind) {
        emitDegenerate();
        return;
    }
    // Near-clip the segment in view space to mirror projectSegmentToScreenClipped().
    if (a_behind) {
        float t = (kMinViewZ - view_a.z) / (view_b.z - view_a.z);
        view_a = mix(view_a, view_b, clamp(t, 0.0, 1.0));
        view_a.z = kMinViewZ;
    } else if (b_behind) {
        float t = (kMinViewZ - view_b.z) / (view_a.z - view_b.z);
        view_b = mix(view_b, view_a, clamp(t, 0.0, 1.0));
        view_b.z = kMinViewZ;
    }

    vec2 panel_pos = pc.viewport_panel.xy;
    vec2 panel_size = max(pc.viewport_panel.zw, vec2(1.0));
    vec2 render_size = max(pc.projection.xy, vec2(1.0));
    float cx = render_size.x * 0.5;
    float cy = render_size.y * 0.5;
    bool ortho = pc.params.w > 0.5;

    vec2 projected_a;
    vec2 projected_b;
    if (ortho) {
        float ortho_scale = max(pc.projection.z, 1.0e-6);
        projected_a = vec2(cx + view_a.x * ortho_scale, cy - view_a.y * ortho_scale);
        projected_b = vec2(cx + view_b.x * ortho_scale, cy - view_b.y * ortho_scale);
    } else {
        float depth_a = max(-view_a.z, 1.0e-6);
        float depth_b = max(-view_b.z, 1.0e-6);
        projected_a = vec2(cx + view_a.x * pc.projection.z / depth_a,
                           cy - view_a.y * pc.projection.w / depth_a);
        projected_b = vec2(cx + view_b.x * pc.projection.z / depth_b,
                           cy - view_b.y * pc.projection.w / depth_b);
    }
    vec2 scale = panel_size / render_size;
    vec2 screen_a = panel_pos + projected_a * scale;
    vec2 screen_b = panel_pos + projected_b * scale;

    float thickness = max(pc.params.z, 1.0);
    float extent = thickness * 0.5 + 2.0;
    vec2 delta = screen_b - screen_a;
    float len = length(delta);
    vec2 dir = (len > 1e-4) ? (delta / len) : vec2(1.0, 0.0);
    vec2 nrm = vec2(-dir.y, dir.x);

    float end_sel = QUAD[corner].x;
    float side = QUAD[corner].y;
    vec2 base = (end_sel < 0.5) ? (screen_a - dir * extent)
                                : (screen_b + dir * extent);
    vec2 screen = base + nrm * (side * extent);

    float depth_a = max(-view_a.z, 0.0);
    float depth_b = max(-view_b.z, 0.0);

    ScreenPos = screen;
    P0 = screen_a;
    P1 = screen_b;
    Color = inst.color;
    Params = vec4(0.0, thickness, 0.0, 1.0);
    ViewDepth = (end_sel < 0.5) ? depth_a : depth_b;

    vec2 ndc = vec2((screen.x - panel_pos.x) / panel_size.x,
                    (screen.y - panel_pos.y) / panel_size.y) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
