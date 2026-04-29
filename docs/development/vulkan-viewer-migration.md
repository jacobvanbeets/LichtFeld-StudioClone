# Vulkan Viewer Migration

This migration replaces the viewer's OpenGL integration with Vulkan in stages. The current viewer is not a thin windowing integration: OpenGL resource IDs are part of the renderer, RmlUi bridge, ImGui bridge, CUDA interop path, split-view cache, thumbnails, and public frame contracts.

## Dependency Baseline

Vulkan is resolved through vcpkg so the Windows and portable builds use the same dependency source:

- `vulkan` provides `Vulkan::Vulkan` and the loader.
- `vulkan-memory-allocator` is available for image and buffer lifetime management.
- `volk` is available if the renderer switches to generated dispatch tables.
- `imgui[vulkan-binding]` is enabled while the existing OpenGL binding remains in place during the transition.

`ENABLE_VULKAN_VIEWER` controls whether CMake resolves and links the Vulkan viewer dependencies. The application probes the Vulkan loader at startup so portable builds report missing loader/runtime problems early in logs.

## Required Code Migration

1. Replace `SDL_WINDOW_OPENGL` and `SDL_GLContext` in `WindowManager` with `SDL_WINDOW_VULKAN`, `SDL_Vulkan_GetInstanceExtensions`, and `SDL_Vulkan_CreateSurface`.
2. Introduce a viewer-owned Vulkan context: instance, surface, physical device selection, device, queues, swapchain, command pools, frame synchronization, and resize handling.
3. Replace `GpuFrame`/`TextureHandle` OpenGL IDs with backend-neutral texture handles carrying Vulkan image/image-view/sampler ownership.
4. Replace CUDA-OpenGL interop with CUDA external memory and semaphore interop for Vulkan images and buffers.
5. Port the screen presentation, split-view compositor, depth compositor, mesh, point cloud, grid, gizmo, text, thumbnail, and environment renderers to Vulkan pipelines and descriptor sets.
6. Replace `ImGui_ImplOpenGL3` with `ImGui_ImplVulkan` and initialize it from the viewer Vulkan context.
7. Replace the copied RmlUi GL3 backend with RmlUi's Vulkan renderer or a project-owned RenderInterface using the viewer command buffer and swapchain.
8. Convert GLSL shaders to SPIR-V at build time and install the compiled shader assets in portable builds.
9. Remove `glad`, `OpenGL`, and `ENABLE_CUDA_GL_INTEROP` once no viewer code includes GL types or calls.

The last step should only happen after the Vulkan path can render the viewport, UI, split view, overlays, readback, and export flows without an OpenGL context.
