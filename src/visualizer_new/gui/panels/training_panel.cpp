/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/training_panel.hpp"
#include "core_new/events.hpp"
#include "core_new/logger.hpp"
#include "core_new/parameters.hpp"
#include "gui/ui_widgets.hpp"
#include "gui/utils/windows_utils.hpp"
#include "theme/theme.hpp"
#include "visualizer_impl.hpp"

#include <chrono>
#include <cstring>
#include <deque>
#include <filesystem>
#include <imgui.h>

#ifdef _WIN32
#include <Windows.h>
#endif

namespace lfs::vis::gui::panels {

    // Helper to convert between old and new parameter types
    // Cannot use memcpy because structures contain non-trivial types (std::string, std::vector)


    // Iteration rate tracking
    struct IterationRateTracker {
        struct Sample {
            int iteration;
            std::chrono::steady_clock::time_point timestamp;
        };

        std::deque<Sample> samples;
        float window_seconds = 5.0f; // Configurable averaging window

        void addSample(int iteration) {
            auto now = std::chrono::steady_clock::now();
            samples.push_back({iteration, now});

            // Remove old samples outside the window
            while (!samples.empty()) {
                auto age = std::chrono::duration_cast<std::chrono::milliseconds>(now - samples.front().timestamp).count() / 1000.0f;
                if (age <= window_seconds) {
                    break;
                }
                samples.pop_front();
            }
        }

        float getIterationsPerSecond() const {
            if (samples.size() < 2) {
                return 0.0f;
            }

            const auto& oldest = samples.front();
            const auto& newest = samples.back();

            int iter_diff = newest.iteration - oldest.iteration;
            auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(newest.timestamp - oldest.timestamp).count() / 1000.0f;

            if (time_diff <= 0.0f) {
                return 0.0f;
            }

            return iter_diff / time_diff;
        }

        void clear() {
            samples.clear();
        }

        void setWindowSeconds(float seconds) {
            window_seconds = seconds;
        }
    };

    namespace {
        constexpr const char* PARAMETER_DIR = "parameter";

        std::filesystem::path get_config_path(const std::string& filename) {
#ifdef _WIN32
            char exe_path_buf[MAX_PATH];
            GetModuleFileNameA(nullptr, exe_path_buf, MAX_PATH);
            std::filesystem::path search_dir = std::filesystem::path(exe_path_buf).parent_path();
#else
            std::filesystem::path search_dir = std::filesystem::read_symlink("/proc/self/exe").parent_path();
#endif
            while (!search_dir.empty()) {
                if (std::filesystem::exists(search_dir / PARAMETER_DIR / filename)) {
                    return search_dir / PARAMETER_DIR / filename;
                }
                const auto parent = search_dir.parent_path();
                if (parent == search_dir) break;
                search_dir = parent;
            }
            return search_dir / PARAMETER_DIR / filename;
        }
    } // namespace

    // Stores base (unscaled) step values for dynamic step scaling
    struct BaseStepValues {
        size_t iterations = 30000;
        size_t start_refine = 500;
        size_t reset_every = 3000;
        size_t stop_refine = 25000;
        size_t refine_every = 100;
        size_t sh_degree_interval = 1000;
        std::vector<size_t> eval_steps = {7000, 30000};
        std::vector<size_t> save_steps = {7000, 30000};

        void extractFrom(const lfs::core::param::OptimizationParameters& params) {
            const float scaler = params.steps_scaler;
            if (scaler > 0.0f) {
                iterations = static_cast<size_t>(params.iterations / scaler);
                start_refine = static_cast<size_t>(params.start_refine / scaler);
                reset_every = static_cast<size_t>(params.reset_every / scaler);
                stop_refine = static_cast<size_t>(params.stop_refine / scaler);
                refine_every = static_cast<size_t>(params.refine_every / scaler);
                sh_degree_interval = static_cast<size_t>(params.sh_degree_interval / scaler);
                eval_steps.clear();
                eval_steps.reserve(params.eval_steps.size());
                for (const auto step : params.eval_steps) {
                    eval_steps.push_back(static_cast<size_t>(step / scaler));
                }
                save_steps.clear();
                save_steps.reserve(params.save_steps.size());
                for (const auto step : params.save_steps) {
                    save_steps.push_back(static_cast<size_t>(step / scaler));
                }
            } else {
                iterations = params.iterations;
                start_refine = params.start_refine;
                reset_every = params.reset_every;
                stop_refine = params.stop_refine;
                refine_every = params.refine_every;
                sh_degree_interval = params.sh_degree_interval;
                eval_steps = params.eval_steps;
                save_steps = params.save_steps;
            }
        }

        void applyTo(lfs::core::param::OptimizationParameters& params, const float new_scaler) const {
            if (new_scaler > 0.0f) {
                params.iterations = static_cast<size_t>(iterations * new_scaler);
                params.start_refine = static_cast<size_t>(start_refine * new_scaler);
                params.reset_every = static_cast<size_t>(reset_every * new_scaler);
                params.stop_refine = static_cast<size_t>(stop_refine * new_scaler);
                params.refine_every = static_cast<size_t>(refine_every * new_scaler);
                params.sh_degree_interval = static_cast<size_t>(sh_degree_interval * new_scaler);
                params.eval_steps.clear();
                params.eval_steps.reserve(eval_steps.size());
                for (const auto step : eval_steps) {
                    params.eval_steps.push_back(static_cast<size_t>(step * new_scaler));
                }
                params.save_steps.clear();
                params.save_steps.reserve(save_steps.size());
                for (const auto step : save_steps) {
                    params.save_steps.push_back(static_cast<size_t>(step * new_scaler));
                }
            } else {
                params.iterations = iterations;
                params.start_refine = start_refine;
                params.reset_every = reset_every;
                params.stop_refine = stop_refine;
                params.refine_every = refine_every;
                params.sh_degree_interval = sh_degree_interval;
                params.eval_steps = eval_steps;
                params.save_steps = save_steps;
            }
            params.steps_scaler = new_scaler;
        }
    };

    // Caches parameters for both strategies to preserve settings when switching
    struct StrategyParamsCache {
        static constexpr const char* MCMC_CONFIG = "mcmc_optimization_params.json";
        static constexpr const char* DEFAULT_CONFIG = "default_optimization_params.json";

        lfs::core::param::OptimizationParameters mcmc_params;
        lfs::core::param::OptimizationParameters default_params;
        BaseStepValues mcmc_base_steps;
        BaseStepValues default_base_steps;
        bool initialized = false;
        std::string last_data_path;

        void initialize(const lfs::core::param::OptimizationParameters& current_params) {
            const bool is_mcmc = (current_params.strategy == "mcmc");

            if (is_mcmc) {
                mcmc_params = current_params;
                mcmc_base_steps.extractFrom(current_params);
                loadAlternateStrategy(default_params, default_base_steps, DEFAULT_CONFIG, "default");
            } else {
                default_params = current_params;
                default_base_steps.extractFrom(current_params);
                loadAlternateStrategy(mcmc_params, mcmc_base_steps, MCMC_CONFIG, "mcmc");
            }
            initialized = true;
        }

        void storeCurrentParams(const lfs::core::param::OptimizationParameters& params) {
            if (params.strategy == "mcmc") {
                mcmc_params = params;
            } else {
                default_params = params;
            }
        }

        lfs::core::param::OptimizationParameters& getParamsForStrategy(const std::string& strategy) {
            return (strategy == "mcmc") ? mcmc_params : default_params;
        }

        BaseStepValues& getBaseStepsForStrategy(const std::string& strategy) {
            return (strategy == "mcmc") ? mcmc_base_steps : default_base_steps;
        }

        void applyStepScaling(lfs::core::param::OptimizationParameters& params, const float new_scaler) {
            getBaseStepsForStrategy(params.strategy).applyTo(params, new_scaler);
        }

    private:
        void loadAlternateStrategy(lfs::core::param::OptimizationParameters& params,
                                   BaseStepValues& base_steps,
                                   const char* config_file,
                                   const char* strategy_name) {
            auto config_path = get_config_path(config_file);
            auto result = lfs::core::param::read_optim_params_from_json(config_path);
            if (result) {
                params = *result;
                base_steps.extractFrom(*result);
                LOG_DEBUG("Loaded {} params from: {}", strategy_name, config_path.string());
            } else {
                params = lfs::core::param::OptimizationParameters{};
                params.strategy = strategy_name;
                base_steps.extractFrom(params);
            }
        }
    };

    void DrawTrainingParameters(const UIContext& ctx) {
        auto* trainer_manager = ctx.viewer->getTrainerManager();
        if (!trainer_manager || !trainer_manager->hasTrainer()) {
            return;
        }

        const auto trainer_state = trainer_manager->getState();
        const bool can_edit = (trainer_state == TrainerManager::State::Ready);

        lfs::core::param::OptimizationParameters opt_params;
        lfs::core::param::DatasetConfig dataset_params;

        if (can_edit) {
            opt_params = trainer_manager->getEditableOptParams();
            dataset_params = trainer_manager->getEditableDatasetParams();
        } else {
            const auto* trainer = trainer_manager->getTrainer();
            if (!trainer) return;
            const auto& params = trainer->getParams();
            opt_params = params.optimization;
            dataset_params = params.dataset;
        }

        // Strategy parameter cache - preserves settings when switching strategies
        static StrategyParamsCache strategy_cache;

        // Initialize or re-initialize cache when project changes
        std::string current_data_path = dataset_params.data_path.string();
        if (!strategy_cache.initialized || strategy_cache.last_data_path != current_data_path) {
            strategy_cache.initialize(opt_params);
            strategy_cache.last_data_path = current_data_path;
        }

        // Track changes separately for optimization and dataset parameters
        bool opt_params_changed = false;
        bool dataset_params_changed = false;

        // Check if masks folder exists in data path
        bool has_masks = false;
        if (!dataset_params.data_path.empty()) {
            static constexpr std::array<const char*, 3> MASK_FOLDERS = {"masks", "mask", "segmentation"};
            for (const auto& folder : MASK_FOLDERS) {
                if (std::filesystem::exists(dataset_params.data_path / folder)) {
                    has_masks = true;
                    break;
                }
            }
        }

        ImGui::PushStyleVar(ImGuiStyleVar_IndentSpacing, 12.0f);
        if (ImGui::BeginTable("DatasetTable", 2, ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 120.0f);
            ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

            // Strategy selector
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("Strategy:");
            ImGui::TableNextColumn();
            if (can_edit) {
                ImGui::PushItemWidth(-1);
                static const char* strategy_labels[] = {"MCMC", "Default"};
                int current_strategy = (opt_params.strategy == "mcmc") ? 0 : 1;
                if (ImGui::Combo("##strategy", &current_strategy, strategy_labels, 2)) {
                    std::string new_strategy = (current_strategy == 0) ? "mcmc" : "default";
                    if (new_strategy != opt_params.strategy) {
                        // Store current parameters before switching
                        strategy_cache.storeCurrentParams(opt_params);

                        // Get cached parameters for the new strategy
                        auto& cached_params = strategy_cache.getParamsForStrategy(new_strategy);

                        // Preserve common parameters that should not change when switching strategies
                        cached_params.iterations = opt_params.iterations;
                        cached_params.sh_degree = opt_params.sh_degree;
                        cached_params.tile_mode = opt_params.tile_mode;
                        cached_params.num_workers = opt_params.num_workers;
                        cached_params.use_bilateral_grid = opt_params.use_bilateral_grid;
                        cached_params.bilateral_grid_X = opt_params.bilateral_grid_X;
                        cached_params.bilateral_grid_Y = opt_params.bilateral_grid_Y;
                        cached_params.bilateral_grid_W = opt_params.bilateral_grid_W;
                        cached_params.bilateral_grid_lr = opt_params.bilateral_grid_lr;
                        cached_params.tv_loss_weight = opt_params.tv_loss_weight;
                        cached_params.mask_mode = opt_params.mask_mode;
                        cached_params.invert_masks = opt_params.invert_masks;
                        cached_params.mask_threshold = opt_params.mask_threshold;
                        cached_params.mask_opacity_penalty_weight = opt_params.mask_opacity_penalty_weight;
                        cached_params.mask_opacity_penalty_power = opt_params.mask_opacity_penalty_power;
                        cached_params.enable_sparsity = opt_params.enable_sparsity;
                        cached_params.sparsify_steps = opt_params.sparsify_steps;
                        cached_params.init_rho = opt_params.init_rho;
                        cached_params.prune_ratio = opt_params.prune_ratio;
                        cached_params.enable_eval = opt_params.enable_eval;
                        cached_params.enable_save_eval_images = opt_params.enable_save_eval_images;
                        cached_params.eval_steps = opt_params.eval_steps;
                        cached_params.save_steps = opt_params.save_steps;
                        cached_params.lambda_dssim = opt_params.lambda_dssim;
                        cached_params.opacity_reg = opt_params.opacity_reg;
                        cached_params.scale_reg = opt_params.scale_reg;
                        cached_params.gut = opt_params.gut;
                        cached_params.bg_modulation = opt_params.bg_modulation;

                        // Switch to cached parameters (strategy-specific values preserved)
                        opt_params = cached_params;
                        opt_params_changed = true;
                    }
                }
                ImGui::PopItemWidth();
            } else {
                ImGui::Text("%s", opt_params.strategy == "mcmc" ? "MCMC" : "Default");
            }

            // Iterations - EDITABLE
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("Iterations:");
            ImGui::TableNextColumn();
            if (can_edit) {
                ImGui::PushItemWidth(-1);
                int iterations = static_cast<int>(opt_params.iterations);
                if (ImGui::InputInt("##iterations", &iterations, 1000, 5000)) {
                    if (iterations > 0 && iterations <= 1000000) {
                        opt_params.iterations = static_cast<size_t>(iterations);
                        opt_params_changed = true;
                    }
                }
                ImGui::PopItemWidth();
            } else {
                ImGui::Text("%zu", opt_params.iterations);
            }

            if (opt_params.strategy == "mcmc") {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Max Gaussians:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::InputInt("##max_cap", &opt_params.max_cap, 10000, 100000)) {
                        if (opt_params.max_cap > 0) {
                            opt_params_changed = true;
                        }
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%d", opt_params.max_cap);
                }
            }

            // SH Degree (dropdown)
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("SH Degree:");
            ImGui::TableNextColumn();
            if (can_edit) {
                ImGui::PushItemWidth(-1);
                static const char* sh_degree_labels[] = {"0", "1", "2", "3"};
                if (ImGui::Combo("##sh_degree", &opt_params.sh_degree, sh_degree_labels, 4)) {
                    opt_params_changed = true;
                }
                ImGui::PopItemWidth();
            } else {
                ImGui::Text("%d", opt_params.sh_degree);
            }

            // Tile Mode
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("Tile Mode:");
            ImGui::TableNextColumn();
            if (can_edit) {
                ImGui::PushItemWidth(-1);
                static const int tile_options[] = {1, 2, 4};
                static const char* tile_labels[] = {"1 (Full)", "2 (Half)", "4 (Quarter)"};
                int current_tile_index = 0;
                for (int i = 0; i < 3; ++i) {
                    if (opt_params.tile_mode == tile_options[i]) {
                        current_tile_index = i;
                    }
                }
                if (ImGui::Combo("##tile_mode", &current_tile_index, tile_labels, 3)) {
                    opt_params.tile_mode = tile_options[current_tile_index];
                    opt_params_changed = true;
                }
                ImGui::PopItemWidth();
            } else {
                ImGui::Text("%d", opt_params.tile_mode);
            }

            // Num Workers
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("Num Workers:");
            ImGui::TableNextColumn();
            if (can_edit) {
                ImGui::PushItemWidth(-1);
                if (ImGui::InputInt("##num_workers", &opt_params.num_workers, 1, 4)) {
                    if (opt_params.num_workers > 0 && opt_params.num_workers <= 64) {
                        opt_params_changed = true;
                    }
                }
                ImGui::PopItemWidth();
            } else {
                ImGui::Text("%d", opt_params.num_workers);
            }

            // Steps Scaler
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("Steps Scaler:");
            ImGui::TableNextColumn();
            if (can_edit) {
                ImGui::PushItemWidth(-1);
                float new_scaler = opt_params.steps_scaler;
                if (ImGui::InputFloat("##steps_scaler", &new_scaler, 0.1f, 0.5f, "%.2f")) {
                    if (new_scaler >= 0.0f) {  // Allow 0 (disabled) or positive values
                        // Apply step scaling to all affected parameters
                        strategy_cache.applyStepScaling(opt_params, new_scaler);
                        opt_params_changed = true;
                    }
                }
                ImGui::PopItemWidth();
                if (opt_params.steps_scaler > 0) {
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "(scaling active)");
                }
            } else {
                ImGui::Text("%.2f", opt_params.steps_scaler);
            }

            // Bilateral Grid Enable
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("Bilateral Grid:");
            ImGui::TableNextColumn();
            if (can_edit) {
                if (ImGui::Checkbox("##use_bilateral_grid", &opt_params.use_bilateral_grid)) {
                    opt_params_changed = true;
                }
            } else {
                ImGui::Text("%s", opt_params.use_bilateral_grid ? "Enabled" : "Disabled");
            }

            // Mask Mode (only editable if masks folder exists)
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("Mask Mode:");
            ImGui::TableNextColumn();
            if (can_edit && has_masks) {
                ImGui::PushItemWidth(-1);
                const char* mask_mode_items[] = {"None", "Segment", "Ignore", "Alpha Consistent"};
                int current_mask_mode = static_cast<int>(opt_params.mask_mode);
                if (ImGui::Combo("##mask_mode", &current_mask_mode, mask_mode_items, IM_ARRAYSIZE(mask_mode_items))) {
                    opt_params.mask_mode = static_cast<lfs::core::param::MaskMode>(current_mask_mode);
                    opt_params_changed = true;
                }
                ImGui::PopItemWidth();
            } else {
                const char* mode_names[] = {"None", "Segment", "Ignore", "Alpha Consistent"};
                ImGui::Text("%s", mode_names[static_cast<int>(opt_params.mask_mode)]);
                if (!has_masks && can_edit) {
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "(no masks)");
                }
            }

            // Enable Sparsity
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("Sparsity:");
            ImGui::TableNextColumn();
            if (can_edit) {
                if (ImGui::Checkbox("##enable_sparsity", &opt_params.enable_sparsity)) {
                    opt_params_changed = true;
                }
            } else {
                ImGui::Text("%s", opt_params.enable_sparsity ? "Enabled" : "Disabled");
            }

            // GUT (Gaussian Unbinding Transform)
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("GUT:");
            ImGui::TableNextColumn();
            if (can_edit) {
                if (ImGui::Checkbox("##gut", &opt_params.gut)) {
                    opt_params_changed = true;
                }
            } else {
                ImGui::Text("%s", opt_params.gut ? "Enabled" : "Disabled");
            }

            // BG Modulation
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("BG Modulation:");
            ImGui::TableNextColumn();
            if (can_edit) {
                if (ImGui::Checkbox("##bg_modulation", &opt_params.bg_modulation)) {
                    opt_params_changed = true;
                }
            } else {
                ImGui::Text("%s", opt_params.bg_modulation ? "Enabled" : "Disabled");
            }

            // Evaluation
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("Evaluation:");
            ImGui::TableNextColumn();
            if (can_edit) {
                if (ImGui::Checkbox("##enable_eval", &opt_params.enable_eval)) {
                    opt_params_changed = true;
                }
            } else {
                ImGui::Text("%s", opt_params.enable_eval ? "Enabled" : "Disabled");
            }
        }
        ImGui::EndTable();

        // Advanced Training Parameters section (as a TreeNode to distinguish from main header)
        ImGui::Spacing();
        if (ImGui::TreeNode("Advanced Training Params")) {

        // Dataset Parameters
        if (ImGui::TreeNode("Dataset")) {
            if (ImGui::BeginTable("DatasetTable", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 120.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Path:");
                ImGui::TableNextColumn();
                ImGui::Text("%s", dataset_params.data_path.filename().string().c_str());

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Images:");
                ImGui::TableNextColumn();
                ImGui::Text("%s", dataset_params.images.c_str());

                // Resize Factor - EDITABLE
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Resize Factor:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    // Available options
                    static const int resize_options[] = {1, 2, 4, 8};
                    static const char* resize_labels[] = {"1", "2", "4", "8"};
                    static int current_index = 0; // default is 1
                    int array_size = IM_ARRAYSIZE(resize_labels);
                    // Set current_index to current value, if needed
                    for (int i = 0; i < array_size; ++i) {
                        if (dataset_params.resize_factor == resize_options[i]) {
                            current_index = i;
                        }
                    }

                    // Draw combo
                    if (ImGui::Combo("##resize_factor", &current_index, resize_labels, array_size)) {
                        dataset_params.resize_factor = resize_options[current_index];
                        dataset_params_changed = true;
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%d", dataset_params.resize_factor);
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Max Width (px):");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::InputInt("##max_width", &dataset_params.max_width, 80, 400)) {
                        if (dataset_params.max_width > 0 && dataset_params.max_width <= 4096) {
                            dataset_params_changed = true;
                        }
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%d", dataset_params.max_width);
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("CPU Cache:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    if (ImGui::Checkbox("##use_cpu_cache", &dataset_params.loading_params.use_cpu_memory)) {
                        dataset_params_changed = true;
                    }
                } else {
                    ImGui::Text("%s", dataset_params.loading_params.use_cpu_memory ? "Enabled" : "Disabled");
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("FS Cache:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    if (ImGui::Checkbox("##use_fs_cache", &dataset_params.loading_params.use_fs_cache)) {
                        dataset_params_changed = true;
                    }
                } else {
                    ImGui::Text("%s", dataset_params.loading_params.use_fs_cache ? "Enabled" : "Disabled");
                }

                // Test Every - EDITABLE (only shown if evaluation is enabled)
                if (opt_params.enable_eval) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Test Every:");
                    ImGui::TableNextColumn();
                    if (can_edit) {
                        ImGui::PushItemWidth(-1);
                        if (ImGui::InputInt("##test_every", &dataset_params.test_every, 100, 500)) {
                            if (dataset_params.test_every > 0 && dataset_params.test_every <= 10000) {
                                dataset_params_changed = true;
                            }
                        }
                        ImGui::PopItemWidth();
                    } else {
                        ImGui::Text("%d", dataset_params.test_every);
                    }
                }

                // Output Folder - EDITABLE with browse button
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Output:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    std::string output_display = dataset_params.output_path.empty()
                        ? "(default)"
                        : dataset_params.output_path.filename().string();
                    ImGui::Text("%s", output_display.c_str());
                    if (!dataset_params.output_path.empty() && ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("%s", dataset_params.output_path.string().c_str());
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Browse##output")) {
                        // Start in output folder if set, otherwise data folder
                        auto start_dir = dataset_params.output_path.empty()
                            ? dataset_params.data_path
                            : dataset_params.output_path;
                        auto selected = SelectFolderDialog("Select Output Folder", start_dir);
                        if (!selected.empty()) {
                            dataset_params.output_path = selected;
                            dataset_params_changed = true;
                        }
                    }
                } else {
                    std::string output_display = dataset_params.output_path.empty()
                        ? "(default)"
                        : dataset_params.output_path.filename().string();
                    ImGui::Text("%s", output_display.c_str());
                }

                ImGui::EndTable();
            }
            ImGui::TreePop();
        }

        // Optimization Parameters
        if (ImGui::TreeNode("Optimization")) {
            if (ImGui::BeginTable("OptimizationTable", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 120.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Strategy:");
                ImGui::TableNextColumn();
                ImGui::Text("%s", opt_params.strategy.c_str());

                // Learning Rates section - ALL EDITABLE
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(theme().palette.text_dim, "Learning Rates:");
                ImGui::TableNextColumn();

                // Position LR
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Position:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::InputFloat("##means_lr", &opt_params.means_lr, 0.000001f, 0.00001f, "%.6f")) {
                        opt_params_changed = true;
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.6f", opt_params.means_lr);
                }

                // SH Coeff LR
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  SH Coeff:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::InputFloat("##shs_lr", &opt_params.shs_lr, 0.0001f, 0.001f, "%.4f")) {
                        opt_params_changed = true;
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.4f", opt_params.shs_lr);
                }

                // Opacity LR
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Opacity:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::InputFloat("##opacity_lr", &opt_params.opacity_lr, 0.001f, 0.01f, "%.4f")) {
                        opt_params_changed = true;
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.4f", opt_params.opacity_lr);
                }

                // Scaling LR
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Scaling:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::InputFloat("##scaling_lr", &opt_params.scaling_lr, 0.0001f, 0.001f, "%.4f")) {
                        opt_params_changed = true;
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.4f", opt_params.scaling_lr);
                }

                // Rotation LR
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Rotation:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::InputFloat("##rotation_lr", &opt_params.rotation_lr, 0.0001f, 0.001f, "%.4f")) {
                        opt_params_changed = true;
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.4f", opt_params.rotation_lr);
                }

                // Refinement section - ALL EDITABLE
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(theme().palette.text_dim, "Refinement:");
                ImGui::TableNextColumn();

                // Refine Every
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Refine Every:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    int refine_every = static_cast<int>(opt_params.refine_every);
                    if (ImGui::InputInt("##refine_every", &refine_every, 10, 100)) {
                        if (refine_every > 0) {
                            opt_params.refine_every = static_cast<size_t>(refine_every);
                            opt_params_changed = true;
                        }
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%zu", opt_params.refine_every);
                }

                // Start Refine
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Start Refine:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    int start_refine = static_cast<int>(opt_params.start_refine);
                    if (ImGui::InputInt("##start_refine", &start_refine, 100, 500)) {
                        if (start_refine >= 0) {
                            opt_params.start_refine = static_cast<size_t>(start_refine);
                            opt_params_changed = true;
                        }
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%zu", opt_params.start_refine);
                }

                // Stop Refine
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Stop Refine:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    int stop_refine = static_cast<int>(opt_params.stop_refine);
                    if (ImGui::InputInt("##stop_refine", &stop_refine, 1000, 5000)) {
                        if (stop_refine >= 0) {
                            opt_params.stop_refine = static_cast<size_t>(stop_refine);
                            opt_params_changed = true;
                        }
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%zu", opt_params.stop_refine);
                }

                // Gradient Threshold
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Gradient Thr:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::InputFloat("##grad_threshold", &opt_params.grad_threshold, 0.000001f, 0.00001f, "%.6f")) {
                        opt_params_changed = true;
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.6f", opt_params.grad_threshold);
                }

                // Reset Every
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  Reset Every:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    int reset_every = static_cast<int>(opt_params.reset_every);
                    if (ImGui::InputInt("##reset_every", &reset_every, 100, 1000)) {
                        if (reset_every >= 0) {
                            opt_params.reset_every = static_cast<size_t>(reset_every);
                            opt_params_changed = true;
                        }
                    }
                    ImGui::PopItemWidth();
                } else if (opt_params.reset_every > 0) {
                    ImGui::Text("%zu", opt_params.reset_every);
                } else {
                    ImGui::Text("Disabled");
                }

                // SH Degree Interval (moved from Initialization)
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("  SH Upgrade Every:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    int sh_deg_interval = static_cast<int>(opt_params.sh_degree_interval);
                    if (ImGui::InputInt("##sh_degree_interval", &sh_deg_interval, 100, 500)) {
                        if (sh_deg_interval > 0) {
                            opt_params.sh_degree_interval = static_cast<size_t>(sh_deg_interval);
                            opt_params_changed = true;
                        }
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%zu", opt_params.sh_degree_interval);
                }

                ImGui::EndTable();
            }
            ImGui::TreePop();
        }

        // Save Steps - FULLY EDITABLE
        if (ImGui::TreeNode("Save Steps")) {
            if (can_edit) {
                // Add new save step
                static int new_step = 1000;
                ImGui::InputInt("New Step", &new_step, 100, 1000);
                ImGui::SameLine();
                if (ImGui::Button("Add")) {
                    if (new_step > 0 && std::find(opt_params.save_steps.begin(),
                                                  opt_params.save_steps.end(),
                                                  new_step) == opt_params.save_steps.end()) {
                        opt_params.save_steps.push_back(new_step);
                        std::sort(opt_params.save_steps.begin(), opt_params.save_steps.end());
                        opt_params_changed = true;
                    }
                }

                ImGui::Separator();

                // List existing save steps with remove buttons
                for (size_t i = 0; i < opt_params.save_steps.size(); ++i) {
                    ImGui::PushID(static_cast<int>(i));

                    int step = static_cast<int>(opt_params.save_steps[i]);
                    ImGui::SetNextItemWidth(100);
                    if (ImGui::InputInt("##step", &step, 0, 0)) {
                        if (step > 0) {
                            opt_params.save_steps[i] = static_cast<size_t>(step);
                            std::sort(opt_params.save_steps.begin(), opt_params.save_steps.end());
                            opt_params_changed = true;
                        }
                    }

                    ImGui::SameLine();
                    if (ImGui::Button("Remove")) {
                        opt_params.save_steps.erase(opt_params.save_steps.begin() + i);
                        opt_params_changed = true;
                    }

                    ImGui::PopID();
                }

                if (opt_params.save_steps.empty()) {
                    ImGui::TextColored(darken(theme().palette.text_dim, 0.15f), "No save steps configured");
                }
            } else {
                // Read-only display
                if (!opt_params.save_steps.empty()) {
                    std::string steps_str;
                    for (size_t i = 0; i < opt_params.save_steps.size(); ++i) {
                        if (i > 0)
                            steps_str += ", ";
                        steps_str += std::to_string(opt_params.save_steps[i]);
                    }
                    ImGui::Text("%s", steps_str.c_str());
                } else {
                    ImGui::TextColored(darken(theme().palette.text_dim, 0.15f), "No save steps");
                }
            }
            ImGui::TreePop();
        }

        // Bilateral Grid Settings (only show when enabled)
        if (opt_params.use_bilateral_grid && ImGui::TreeNode("Bilateral Grid Settings")) {
            if (ImGui::BeginTable("BilateralTable", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 140.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                // Grid X
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Grid X:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::InputInt("##bilateral_grid_X", &opt_params.bilateral_grid_X, 1, 4)) {
                        if (opt_params.bilateral_grid_X > 0) {
                            opt_params_changed = true;
                        }
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%d", opt_params.bilateral_grid_X);
                }

                // Grid Y
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Grid Y:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::InputInt("##bilateral_grid_Y", &opt_params.bilateral_grid_Y, 1, 4)) {
                        if (opt_params.bilateral_grid_Y > 0) {
                            opt_params_changed = true;
                        }
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%d", opt_params.bilateral_grid_Y);
                }

                // Grid W (luma bins)
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Grid W:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::InputInt("##bilateral_grid_W", &opt_params.bilateral_grid_W, 1, 2)) {
                        if (opt_params.bilateral_grid_W > 0) {
                            opt_params_changed = true;
                        }
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%d", opt_params.bilateral_grid_W);
                }

                // Learning Rate
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Learning Rate:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::InputFloat("##bilateral_grid_lr", &opt_params.bilateral_grid_lr, 0.0001f, 0.001f, "%.5f")) {
                        opt_params_changed = true;
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.5f", opt_params.bilateral_grid_lr);
                }

                ImGui::EndTable();
            }
            ImGui::TreePop();
        }

        // Mask Settings (only show when mask mode is not None)
        if (opt_params.mask_mode != lfs::core::param::MaskMode::None && ImGui::TreeNode("Mask Settings")) {
            if (ImGui::BeginTable("MaskTable", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 140.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                // Invert Masks
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Invert Masks:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    if (ImGui::Checkbox("##invert_masks", &opt_params.invert_masks)) {
                        opt_params_changed = true;
                    }
                } else {
                    ImGui::Text("%s", opt_params.invert_masks ? "Yes" : "No");
                }

                // Mask Threshold
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Threshold:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::SliderFloat("##mask_threshold", &opt_params.mask_threshold, 0.0f, 1.0f, "%.2f")) {
                        opt_params_changed = true;
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.2f", opt_params.mask_threshold);
                }

                // Segment mode specific params
                if (opt_params.mask_mode == lfs::core::param::MaskMode::Segment) {
                    // Opacity Penalty Weight
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Penalty Weight:");
                    ImGui::TableNextColumn();
                    if (can_edit) {
                        ImGui::PushItemWidth(-1);
                        if (ImGui::InputFloat("##mask_penalty_weight", &opt_params.mask_opacity_penalty_weight, 0.1f, 0.5f, "%.2f")) {
                            opt_params_changed = true;
                        }
                        ImGui::PopItemWidth();
                    } else {
                        ImGui::Text("%.2f", opt_params.mask_opacity_penalty_weight);
                    }

                    // Opacity Penalty Power
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Penalty Power:");
                    ImGui::TableNextColumn();
                    if (can_edit) {
                        ImGui::PushItemWidth(-1);
                        if (ImGui::InputFloat("##mask_penalty_power", &opt_params.mask_opacity_penalty_power, 0.5f, 1.0f, "%.1f")) {
                            opt_params_changed = true;
                        }
                        ImGui::PopItemWidth();
                    } else {
                        ImGui::Text("%.1f", opt_params.mask_opacity_penalty_power);
                    }
                }

                ImGui::EndTable();
            }
            ImGui::TreePop();
        }

        // Evaluation Settings (only show when evaluation is enabled)
        if (opt_params.enable_eval && ImGui::TreeNode("Evaluation Settings")) {
            if (ImGui::BeginTable("EvalTable", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 140.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                // Save Eval Images
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Save Images:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    if (ImGui::Checkbox("##enable_save_eval_images", &opt_params.enable_save_eval_images)) {
                        opt_params_changed = true;
                    }
                } else {
                    ImGui::Text("%s", opt_params.enable_save_eval_images ? "Yes" : "No");
                }

                ImGui::EndTable();
            }

            // Eval Steps
            {
                ImGui::Separator();
                ImGui::Text("Evaluation Steps:");
                if (can_edit) {
                    // Add new eval step
                    static int new_eval_step = 7000;
                    ImGui::InputInt("New Eval Step", &new_eval_step, 1000, 5000);
                    ImGui::SameLine();
                    if (ImGui::Button("Add##eval")) {
                        if (new_eval_step > 0 && std::find(opt_params.eval_steps.begin(),
                                                          opt_params.eval_steps.end(),
                                                          new_eval_step) == opt_params.eval_steps.end()) {
                            opt_params.eval_steps.push_back(new_eval_step);
                            std::sort(opt_params.eval_steps.begin(), opt_params.eval_steps.end());
                            opt_params_changed = true;
                        }
                    }

                    // List existing eval steps with remove buttons
                    for (size_t i = 0; i < opt_params.eval_steps.size(); ++i) {
                        ImGui::PushID(static_cast<int>(i + 1000)); // Offset ID to avoid conflicts

                        int step = static_cast<int>(opt_params.eval_steps[i]);
                        ImGui::SetNextItemWidth(100);
                        if (ImGui::InputInt("##eval_step", &step, 0, 0)) {
                            if (step > 0) {
                                opt_params.eval_steps[i] = static_cast<size_t>(step);
                                std::sort(opt_params.eval_steps.begin(), opt_params.eval_steps.end());
                                opt_params_changed = true;
                            }
                        }

                        ImGui::SameLine();
                        if (ImGui::Button("Remove##eval")) {
                            opt_params.eval_steps.erase(opt_params.eval_steps.begin() + i);
                            opt_params_changed = true;
                        }

                        ImGui::PopID();
                    }

                    if (opt_params.eval_steps.empty()) {
                        ImGui::TextColored(darken(theme().palette.text_dim, 0.15f), "No eval steps configured");
                    }
                } else {
                    // Read-only display
                    if (!opt_params.eval_steps.empty()) {
                        std::string steps_str;
                        for (size_t i = 0; i < opt_params.eval_steps.size(); ++i) {
                            if (i > 0)
                                steps_str += ", ";
                            steps_str += std::to_string(opt_params.eval_steps[i]);
                        }
                        ImGui::Text("%s", steps_str.c_str());
                    } else {
                        ImGui::TextColored(darken(theme().palette.text_dim, 0.15f), "No eval steps");
                    }
                }
            }
            ImGui::TreePop();
        }

        // Loss Parameters section
        if (ImGui::TreeNode("Loss Parameters")) {
            if (ImGui::BeginTable("LossTable", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 140.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                // Lambda DSSIM
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Lambda DSSIM:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::SliderFloat("##lambda_dssim", &opt_params.lambda_dssim, 0.0f, 1.0f, "%.2f")) {
                        opt_params_changed = true;
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.2f", opt_params.lambda_dssim);
                }

                // Opacity Regularization
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Opacity Reg:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::InputFloat("##opacity_reg", &opt_params.opacity_reg, 0.001f, 0.01f, "%.4f")) {
                        opt_params_changed = true;
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.4f", opt_params.opacity_reg);
                }

                // Scale Regularization
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Scale Reg:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::InputFloat("##scale_reg", &opt_params.scale_reg, 0.001f, 0.01f, "%.4f")) {
                        opt_params_changed = true;
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.4f", opt_params.scale_reg);
                }

                // TV Loss Weight (for bilateral grid)
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("TV Loss Weight:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::InputFloat("##tv_loss_weight", &opt_params.tv_loss_weight, 1.0f, 5.0f, "%.1f")) {
                        opt_params_changed = true;
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.1f", opt_params.tv_loss_weight);
                }

                ImGui::EndTable();
            }
            ImGui::TreePop();
        }

        // Initialization Parameters
        if (ImGui::TreeNode("Initialization")) {
            if (ImGui::BeginTable("InitTable", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 140.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                // Init Opacity
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Init Opacity:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::SliderFloat("##init_opacity", &opt_params.init_opacity, 0.01f, 1.0f, "%.2f")) {
                        opt_params_changed = true;
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.2f", opt_params.init_opacity);
                }

                // Init Scaling
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Init Scaling:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::InputFloat("##init_scaling", &opt_params.init_scaling, 0.01f, 0.1f, "%.3f")) {
                        opt_params_changed = true;
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.3f", opt_params.init_scaling);
                }

                // Random initialization checkbox
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Random Init:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    if (ImGui::Checkbox("##random", &opt_params.random)) {
                        opt_params_changed = true;
                    }
                } else {
                    ImGui::Text("%s", opt_params.random ? "Yes" : "No");
                }

                // Random init parameters (only show if random is enabled)
                if (opt_params.random) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("  Num Points:");
                    ImGui::TableNextColumn();
                    if (can_edit) {
                        ImGui::PushItemWidth(-1);
                        if (ImGui::InputInt("##init_num_pts", &opt_params.init_num_pts, 10000, 50000)) {
                            if (opt_params.init_num_pts > 0) {
                                opt_params_changed = true;
                            }
                        }
                        ImGui::PopItemWidth();
                    } else {
                        ImGui::Text("%d", opt_params.init_num_pts);
                    }

                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("  Extent:");
                    ImGui::TableNextColumn();
                    if (can_edit) {
                        ImGui::PushItemWidth(-1);
                        if (ImGui::InputFloat("##init_extent", &opt_params.init_extent, 0.5f, 1.0f, "%.1f")) {
                            if (opt_params.init_extent > 0.0f) {
                                opt_params_changed = true;
                            }
                        }
                        ImGui::PopItemWidth();
                    } else {
                        ImGui::Text("%.1f", opt_params.init_extent);
                    }
                }

                ImGui::EndTable();
            }
            ImGui::TreePop();
        }

        // Pruning/Growing Thresholds (for default strategy)
        if (opt_params.strategy == "default" && ImGui::TreeNode("Pruning/Growing")) {
            if (ImGui::BeginTable("PruneTable", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 140.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                // Min Opacity
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Min Opacity:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::InputFloat("##min_opacity", &opt_params.min_opacity, 0.001f, 0.01f, "%.4f")) {
                        opt_params_changed = true;
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.4f", opt_params.min_opacity);
                }

                // Prune Opacity
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Prune Opacity:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::InputFloat("##prune_opacity", &opt_params.prune_opacity, 0.001f, 0.01f, "%.4f")) {
                        opt_params_changed = true;
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.4f", opt_params.prune_opacity);
                }

                // Grow Scale 3D
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Grow Scale 3D:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::InputFloat("##grow_scale3d", &opt_params.grow_scale3d, 0.001f, 0.01f, "%.4f")) {
                        opt_params_changed = true;
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.4f", opt_params.grow_scale3d);
                }

                // Grow Scale 2D
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Grow Scale 2D:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::InputFloat("##grow_scale2d", &opt_params.grow_scale2d, 0.001f, 0.01f, "%.4f")) {
                        opt_params_changed = true;
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.4f", opt_params.grow_scale2d);
                }

                // Prune Scale 3D
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Prune Scale 3D:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::InputFloat("##prune_scale3d", &opt_params.prune_scale3d, 0.01f, 0.05f, "%.3f")) {
                        opt_params_changed = true;
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.3f", opt_params.prune_scale3d);
                }

                // Prune Scale 2D
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Prune Scale 2D:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::InputFloat("##prune_scale2d", &opt_params.prune_scale2d, 0.01f, 0.05f, "%.3f")) {
                        opt_params_changed = true;
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.3f", opt_params.prune_scale2d);
                }

                // Pause Refine After Reset
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Pause After Reset:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    int pause_refine = static_cast<int>(opt_params.pause_refine_after_reset);
                    if (ImGui::InputInt("##pause_refine_after_reset", &pause_refine, 10, 100)) {
                        if (pause_refine >= 0) {
                            opt_params.pause_refine_after_reset = static_cast<size_t>(pause_refine);
                            opt_params_changed = true;
                        }
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%zu", opt_params.pause_refine_after_reset);
                }

                // Revised Opacity
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Revised Opacity:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    if (ImGui::Checkbox("##revised_opacity", &opt_params.revised_opacity)) {
                        opt_params_changed = true;
                    }
                } else {
                    ImGui::Text("%s", opt_params.revised_opacity ? "Yes" : "No");
                }

                ImGui::EndTable();
            }
            ImGui::TreePop();
        }

        // Sparsity Settings (only show when enabled)
        if (opt_params.enable_sparsity && ImGui::TreeNode("Sparsity Settings")) {
            if (ImGui::BeginTable("SparsityTable", 2, ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 140.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                // Sparsify Steps
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Sparsify Steps:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::InputInt("##sparsify_steps", &opt_params.sparsify_steps, 1000, 5000)) {
                        if (opt_params.sparsify_steps > 0) {
                            opt_params_changed = true;
                        }
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%d", opt_params.sparsify_steps);
                }

                // Init Rho
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Init Rho:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::InputFloat("##init_rho", &opt_params.init_rho, 0.0001f, 0.001f, "%.5f")) {
                        opt_params_changed = true;
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.5f", opt_params.init_rho);
                }

                // Prune Ratio
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Prune Ratio:");
                ImGui::TableNextColumn();
                if (can_edit) {
                    ImGui::PushItemWidth(-1);
                    if (ImGui::SliderFloat("##prune_ratio", &opt_params.prune_ratio, 0.0f, 1.0f, "%.2f")) {
                        opt_params_changed = true;
                    }
                    ImGui::PopItemWidth();
                } else {
                    ImGui::Text("%.2f", opt_params.prune_ratio);
                }

                ImGui::EndTable();
            }
            ImGui::TreePop();
        }

        ImGui::TreePop();
        } // End Advanced Training Params

        if (can_edit) {
            if (opt_params_changed) trainer_manager->getEditableOptParams() = opt_params;
            if (dataset_params_changed) trainer_manager->getEditableDatasetParams() = dataset_params;
        }

        ImGui::PopStyleVar();
    }

    void DrawTrainingControls(const UIContext& ctx) {
        ImGui::Text("Training Control");
        ImGui::Separator();

        auto& state = TrainingPanelState::getInstance();

        // Direct call to TrainerManager - no state duplication
        auto* trainer_manager = ctx.viewer->getTrainerManager();
        if (!trainer_manager || !trainer_manager->hasTrainer()) {
            ImGui::TextColored(darken(theme().palette.text_dim, 0.15f), "No trainer loaded");
            return;
        }

        // Get state directly from the single source of truth
        auto trainer_state = trainer_manager->getState();
        int current_iteration = trainer_manager->getCurrentIteration();

        const auto& t = theme();
        // Render controls based on trainer state
        switch (trainer_state) {
        case TrainerManager::State::Idle:
            ImGui::TextColored(darken(t.palette.text_dim, 0.15f), "No trainer loaded");
            break;

        case TrainerManager::State::Ready:
            ImGui::PushStyleColor(ImGuiCol_Button, darken(t.palette.success, 0.3f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, darken(t.palette.success, 0.15f));
            {
                const char* const label = current_iteration > 0 ? "Resume Training" : "Start Training";
                if (ImGui::Button(label, ImVec2(-1, 0))) {
                    lfs::core::events::cmd::StartTraining{}.emit();
                }
            }
            ImGui::PopStyleColor(2);

            if (current_iteration > 0) {
                ImGui::PushStyleColor(ImGuiCol_Button, darken(t.palette.secondary, 0.2f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, darken(t.palette.secondary, 0.05f));
                if (ImGui::Button("Reset Training", ImVec2(-1, 0))) {
                    lfs::core::events::cmd::ResetTraining{}.emit();
                }
                ImGui::PopStyleColor(2);
            }

            break;

        case TrainerManager::State::Running:
            ImGui::PushStyleColor(ImGuiCol_Button, darken(t.palette.warning, 0.2f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, darken(t.palette.warning, 0.05f));
            if (ImGui::Button("Pause", ImVec2(-1, 0))) {
                lfs::core::events::cmd::PauseTraining{}.emit();
            }
            ImGui::PopStyleColor(2);
            break;

        case TrainerManager::State::Paused:
            ImGui::PushStyleColor(ImGuiCol_Button, darken(t.palette.success, 0.3f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, darken(t.palette.success, 0.15f));
            if (ImGui::Button("Resume", ImVec2(-1, 0))) {
                lfs::core::events::cmd::ResumeTraining{}.emit();
            }
            ImGui::PopStyleColor(2);

            // Reset button
            ImGui::PushStyleColor(ImGuiCol_Button, darken(t.palette.secondary, 0.2f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, darken(t.palette.secondary, 0.05f));
            if (ImGui::Button("Reset Training", ImVec2(-1, 0))) {
                lfs::core::events::cmd::ResetTraining{}.emit();
            }
            ImGui::PopStyleColor(2);

            ImGui::PushStyleColor(ImGuiCol_Button, darken(t.palette.error, 0.3f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, darken(t.palette.error, 0.15f));
            if (ImGui::Button("Stop Permanently", ImVec2(-1, 0))) {
                lfs::core::events::cmd::StopTraining{}.emit();
            }
            ImGui::PopStyleColor(2);
            break;

        case TrainerManager::State::Completed:
            ImGui::TextColored(t.palette.success, "Training Complete!");

            // Reset button for completed state
            ImGui::PushStyleColor(ImGuiCol_Button, darken(t.palette.secondary, 0.2f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, darken(t.palette.secondary, 0.05f));
            if (ImGui::Button("Reset for New Training", ImVec2(-1, 0))) {
                lfs::core::events::cmd::ResetTraining{}.emit();
            }
            ImGui::PopStyleColor(2);

            break;

        case TrainerManager::State::Error:
            ImGui::TextColored(t.palette.error, "Training Error!");
            {
                auto error_msg = trainer_manager->getLastError();
                if (!error_msg.empty()) {
                    ImGui::TextWrapped("%s", error_msg.c_str());
                }
                // Reset button
                ImGui::PushStyleColor(ImGuiCol_Button, darken(t.palette.secondary, 0.2f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, darken(t.palette.secondary, 0.05f));
                if (ImGui::Button("Reset Training", ImVec2(-1, 0))) {
                    lfs::core::events::cmd::ResetTraining{}.emit();
                }
                ImGui::PopStyleColor(2);
            }
            break;

        case TrainerManager::State::Stopping:
            ImGui::TextColored(t.palette.text_dim, "Stopping...");
            break;
        }

        // Save checkpoint button (available during training)
        if (trainer_state == TrainerManager::State::Running ||
            trainer_state == TrainerManager::State::Paused) {

            ImGui::PushStyleColor(ImGuiCol_Button, darken(t.palette.info, 0.3f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, darken(t.palette.info, 0.15f));
            if (ImGui::Button("Save Checkpoint", ImVec2(-1, 0))) {
                lfs::core::events::cmd::SaveCheckpoint{}.emit();
                state.save_in_progress = true;
                state.save_start_time = std::chrono::steady_clock::now();
            }
            ImGui::PopStyleColor(2);
        }

        ImGui::Separator();
        if (ImGui::CollapsingHeader("Basic Training Params", ImGuiTreeNodeFlags_DefaultOpen)) {
            DrawTrainingParameters(ctx);
        }

        // Save feedback
        if (state.save_in_progress) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                               now - state.save_start_time)
                               .count();
            if (elapsed < 2000) {
                ImGui::TextColored(t.palette.success, "Checkpoint saved!");
            } else {
                state.save_in_progress = false;
            }
        }

        // Status display
        ImGui::Separator();

        // Helper to convert state to string
        const char* state_str = "Unknown";
        switch (trainer_state) {
        case TrainerManager::State::Idle: state_str = "Idle"; break;
        case TrainerManager::State::Ready: state_str = "Ready"; break;
        case TrainerManager::State::Running: state_str = "Running"; break;
        case TrainerManager::State::Paused: state_str = "Paused"; break;
        case TrainerManager::State::Stopping: state_str = "Stopping"; break;
        case TrainerManager::State::Completed: state_str = "Completed"; break;
        case TrainerManager::State::Error: state_str = "Error"; break;
        }

        // Static tracker instance
        static IterationRateTracker g_iter_rate_tracker;

        ImGui::Text("Status: %s", state_str);
        // Update iteration rate tracker
        g_iter_rate_tracker.addSample(current_iteration);
        // Get iteration rate
        float iters_per_sec = g_iter_rate_tracker.getIterationsPerSecond();
        iters_per_sec = iters_per_sec > 0.0f ? iters_per_sec : 0.0f;

        // Display iteration with rate
        ImGui::Text("Iteration: %d (%.1f iters/sec)", current_iteration, iters_per_sec);

        int num_splats = trainer_manager->getNumSplats();
        ImGui::Text("num Splats: %d", num_splats);
    }

} // namespace lfs::vis::gui::panels
