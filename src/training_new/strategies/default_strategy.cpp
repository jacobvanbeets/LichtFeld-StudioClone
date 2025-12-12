/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "default_strategy.hpp"
#include "core_new/logger.hpp"
#include "core_new/parameters.hpp"
#include "optimizer/render_output.hpp"
#include "strategy_utils.hpp"
#include "kernels/densification_kernels.hpp"

namespace lfs::training {

    DefaultStrategy::DefaultStrategy(lfs::core::SplatData& splat_data) : _splat_data(&splat_data) {}

    void DefaultStrategy::initialize(const lfs::core::param::OptimizationParameters& optimParams) {
        _params = std::make_unique<const lfs::core::param::OptimizationParameters>(optimParams);

        initialize_gaussians(*_splat_data, _params->max_cap);

        _optimizer = create_optimizer(*_splat_data, *_params);
        _optimizer->allocate_gradients(_params->max_cap > 0 ? static_cast<size_t>(_params->max_cap) : 0);
        _scheduler = create_scheduler(*_params, *_optimizer);

        // Initialize densification info: [2, N] tensor for tracking gradients
        _splat_data->_densification_info = lfs::core::Tensor::zeros(
            {2, static_cast<size_t>(_splat_data->size())},
            _splat_data->means().device());
    }

    bool DefaultStrategy::is_refining(int iter) const {
        return (iter > _params->start_refine &&
                iter % _params->refine_every == 0 &&
                iter % _params->reset_every >= _params->pause_refine_after_reset);
    }

    void DefaultStrategy::remove_gaussians(const lfs::core::Tensor& mask) {
        int mask_sum = mask.to(lfs::core::DataType::Int32).sum().template item<int>();

        if (mask_sum == 0) {
            LOG_DEBUG("No Gaussians to remove");
            return;
        }

        LOG_DEBUG("Removing {} Gaussians", mask_sum);
        remove(mask);
    }

    void DefaultStrategy::duplicate(const lfs::core::Tensor& is_duplicated) {
        const lfs::core::Tensor sampled_idxs = is_duplicated.nonzero().squeeze(-1);
        const int num_duplicated = sampled_idxs.shape()[0];

        if (num_duplicated == 0) {
            return;  // Nothing to duplicate
        }

        // Gather parameters for selected Gaussians to duplicate
        LOG_DEBUG("duplicate(): sh0 shape before index_select: {}", _splat_data->sh0().shape().str());
        LOG_DEBUG("duplicate(): shN shape before index_select: {}", _splat_data->shN().shape().str());

        auto pos_selected = _splat_data->means().index_select(0, sampled_idxs).contiguous();
        auto rot_selected = _splat_data->rotation_raw().index_select(0, sampled_idxs).contiguous();
        auto scale_selected = _splat_data->scaling_raw().index_select(0, sampled_idxs).contiguous();
        auto sh0_selected = _splat_data->sh0().index_select(0, sampled_idxs).contiguous();
        auto shN_selected = _splat_data->shN().index_select(0, sampled_idxs).contiguous();
        auto op_selected = _splat_data->opacity_raw().index_select(0, sampled_idxs).contiguous();

        LOG_DEBUG("duplicate(): sh0_selected shape after index_select: {}", sh0_selected.shape().str());
        LOG_DEBUG("duplicate(): shN_selected shape after index_select: {}", shN_selected.shape().str());

        // Concatenate parameters directly in SplatData
        LOG_DEBUG("duplicate(): About to cat means: existing={}, selected={}", _splat_data->means().shape().str(), pos_selected.shape().str());
        _splat_data->means() = _splat_data->means().cat(pos_selected, 0);

        LOG_DEBUG("duplicate(): About to cat rotation: existing={}, selected={}", _splat_data->rotation_raw().shape().str(), rot_selected.shape().str());
        _splat_data->rotation_raw() = _splat_data->rotation_raw().cat(rot_selected, 0);

        LOG_DEBUG("duplicate(): About to cat scaling: existing={}, selected={}", _splat_data->scaling_raw().shape().str(), scale_selected.shape().str());
        _splat_data->scaling_raw() = _splat_data->scaling_raw().cat(scale_selected, 0);

        LOG_ERROR("duplicate(): About to cat sh0: existing={}, selected={}", _splat_data->sh0().shape().str(), sh0_selected.shape().str());
        _splat_data->sh0() = _splat_data->sh0().cat(sh0_selected, 0);

        LOG_ERROR("duplicate(): About to cat shN: existing={}, selected={}", _splat_data->shN().shape().str(), shN_selected.shape().str());
        _splat_data->shN() = _splat_data->shN().cat(shN_selected, 0);

        LOG_DEBUG("duplicate(): About to cat opacity: existing={}, selected={}", _splat_data->opacity_raw().shape().str(), op_selected.shape().str());
        _splat_data->opacity_raw() = _splat_data->opacity_raw().cat(op_selected, 0);

        // Update optimizer states for duplicated Gaussians
        auto update_optimizer_state = [&](ParamType param_type, const char* name) {
            const auto* state = _optimizer->get_state(param_type);
            if (!state) return;

            const auto& grad_shape = state->grad.shape();
            std::vector<size_t> zero_dims = {static_cast<size_t>(num_duplicated)};
            for (size_t i = 1; i < grad_shape.rank(); ++i) {
                zero_dims.push_back(grad_shape[i]);
            }
            auto zero_grads = lfs::core::Tensor::zeros(lfs::core::TensorShape(zero_dims), state->grad.device());

            AdamParamState new_state = *state;
            new_state.exp_avg = state->exp_avg.cat(state->exp_avg.index_select(0, sampled_idxs), 0);
            new_state.exp_avg_sq = state->exp_avg_sq.cat(state->exp_avg_sq.index_select(0, sampled_idxs), 0);
            new_state.grad = state->grad.cat(zero_grads, 0);
            new_state.size = state->size + num_duplicated;
            _optimizer->set_state(param_type, new_state);
            LOG_DEBUG("duplicate(): {} size {} -> {}", name, state->size, new_state.size);
        };

        update_optimizer_state(ParamType::Means, "means");
        update_optimizer_state(ParamType::Rotation, "rotation");
        update_optimizer_state(ParamType::Scaling, "scaling");
        update_optimizer_state(ParamType::Sh0, "sh0");
        update_optimizer_state(ParamType::ShN, "shN");
        update_optimizer_state(ParamType::Opacity, "opacity");
    }

    void DefaultStrategy::split(const lfs::core::Tensor& is_split) {
        const lfs::core::Tensor split_idxs = is_split.nonzero().squeeze(-1);
        const lfs::core::Tensor keep_idxs = is_split.logical_not().nonzero().squeeze(-1);

        const int N = _splat_data->size();
        const int num_split = split_idxs.shape()[0];
        const int num_keep = keep_idxs.shape()[0];

        if (num_split == 0) {
            return;  // Nothing to split
        }

        constexpr int split_size = 2;

        // Get SH dimensions - total elements per Gaussian (coeffs * channels)
        // shN has shape [N, num_coeffs, 3], so total elements = num_coeffs * 3
        const int shN_coeffs = _splat_data->shN().shape()[1];
        const int shN_channels = _splat_data->shN().shape()[2];
        const int shN_dim = shN_coeffs * shN_channels;  // Total elements per Gaussian

        // Generate random noise [2, num_split, 3]
        const lfs::core::Tensor random_noise = lfs::core::Tensor::randn(
            {split_size, static_cast<size_t>(num_split), 3},
            _splat_data->sh0().device());

        // Allocate output tensors [num_keep + num_split*2, ...]
        const int out_size = num_keep + num_split * split_size;
        auto positions_out = lfs::core::Tensor::empty({static_cast<size_t>(out_size), 3}, _splat_data->means().device());
        auto rotations_out = lfs::core::Tensor::empty({static_cast<size_t>(out_size), 4}, _splat_data->rotation_raw().device());
        auto scales_out = lfs::core::Tensor::empty({static_cast<size_t>(out_size), 3}, _splat_data->scaling_raw().device());
        // CUDA kernel produces 2D outputs, we'll reshape to 3D after
        auto sh0_out_2d = lfs::core::Tensor::empty({static_cast<size_t>(out_size), 3}, _splat_data->sh0().device());
        auto shN_out_2d = lfs::core::Tensor::empty({static_cast<size_t>(out_size), static_cast<size_t>(shN_dim)}, _splat_data->shN().device());
        auto opacities_out = lfs::core::Tensor::empty({static_cast<size_t>(out_size), 1}, _splat_data->opacity_raw().device());

        // Call custom CUDA kernel (outputs sh0 and shN separately - NO slice/contiguous overhead!)
        kernels::launch_split_gaussians(
            _splat_data->means().ptr<float>(),
            _splat_data->rotation_raw().ptr<float>(),
            _splat_data->scaling_raw().ptr<float>(),
            _splat_data->sh0().ptr<float>(),
            _splat_data->shN().ptr<float>(),
            _splat_data->opacity_raw().ptr<float>(),
            positions_out.ptr<float>(),
            rotations_out.ptr<float>(),
            scales_out.ptr<float>(),
            sh0_out_2d.ptr<float>(),
            shN_out_2d.ptr<float>(),
            opacities_out.ptr<float>(),
            split_idxs.ptr<int64_t>(),
            keep_idxs.ptr<int64_t>(),
            random_noise.ptr<float>(),
            N,
            num_split,
            num_keep,
            shN_dim,
            _params->revised_opacity,
            nullptr  // default stream
        );

        // Reshape sh0 and shN from flat 1D (per Gaussian) to original 3D structure
        // sh0: [N, 3] -> [N, 1, 3] (3 elements = 1 coeff * 3 channels)
        // shN: [N, 45] -> [N, 15, 3] (45 elements = 15 coeffs * 3 channels)
        auto sh0_out = sh0_out_2d.reshape({out_size, 1, 3});
        auto shN_out = shN_out_2d.reshape({out_size, shN_coeffs, shN_channels});

        // Update SplatData with new tensors (already contiguous from kernel!)
        _splat_data->means() = positions_out;
        _splat_data->rotation_raw() = rotations_out;
        _splat_data->scaling_raw() = scales_out;
        _splat_data->sh0() = sh0_out;
        _splat_data->shN() = shN_out;
        _splat_data->opacity_raw() = opacities_out.squeeze(-1);

        // Update optimizer states for split Gaussians
        auto update_optimizer_state = [&](ParamType param_type) {
            const auto* state = _optimizer->get_state(param_type);
            if (!state) return;

            const size_t n_new = num_split * split_size;

            auto make_zeros = [&](const lfs::core::Tensor& t) {
                const auto& shape = t.shape();
                std::vector<size_t> dims = {n_new};
                for (size_t i = 1; i < shape.rank(); i++) dims.push_back(shape[i]);
                return lfs::core::Tensor::zeros(lfs::core::TensorShape(dims), t.device(), t.dtype());
            };

            AdamParamState new_state = *state;
            new_state.exp_avg = state->exp_avg.index_select(0, keep_idxs).cat(make_zeros(state->exp_avg), 0);
            new_state.exp_avg_sq = state->exp_avg_sq.index_select(0, keep_idxs).cat(make_zeros(state->exp_avg_sq), 0);
            new_state.grad = state->grad.index_select(0, keep_idxs).cat(make_zeros(state->grad), 0);
            new_state.size = num_keep + n_new;
            _optimizer->set_state(param_type, new_state);
        };

        update_optimizer_state(ParamType::Means);
        update_optimizer_state(ParamType::Rotation);
        update_optimizer_state(ParamType::Scaling);
        update_optimizer_state(ParamType::Sh0);
        update_optimizer_state(ParamType::ShN);
        update_optimizer_state(ParamType::Opacity);
    }

    void DefaultStrategy::grow_gs(int iter) {
        lfs::core::Tensor numer = _splat_data->_densification_info[1];
        lfs::core::Tensor denom = _splat_data->_densification_info[0];
        const lfs::core::Tensor grads = numer / denom.clamp_min(1.0f);

        const lfs::core::Tensor is_grad_high = grads > _params->grad_threshold;

        // Get max along last dimension
        const lfs::core::Tensor max_values = _splat_data->get_scaling().max(-1, false);
        const lfs::core::Tensor is_small = max_values <= _params->grow_scale3d * _splat_data->get_scene_scale();
        const lfs::core::Tensor is_duplicated = is_grad_high.logical_and(is_small);

        const auto num_duplicates = static_cast<int64_t>(is_duplicated.sum_scalar());

        const lfs::core::Tensor is_large = is_small.logical_not();
        lfs::core::Tensor is_split = is_grad_high.logical_and(is_large);
        const auto num_split = static_cast<int64_t>(is_split.sum_scalar());

        // First duplicate
        if (num_duplicates > 0) {
            duplicate(is_duplicated);
        }

        // New Gaussians added by duplication will not be split
        auto zeros_to_concat = lfs::core::Tensor::zeros_bool({static_cast<size_t>(num_duplicates)}, is_split.device());
        is_split = is_split.cat(zeros_to_concat, 0);

        if (num_split > 0) {
            split(is_split);
        }
    }

    void DefaultStrategy::remove(const lfs::core::Tensor& is_prune) {
        const lfs::core::Tensor sampled_idxs = is_prune.logical_not().nonzero().squeeze(-1);

        const auto param_fn = [&sampled_idxs](const int i, const lfs::core::Tensor& param) {
            return param.index_select(0, sampled_idxs);
        };

        const auto optimizer_fn = [&sampled_idxs](
            AdamParamState& state,
            const lfs::core::Tensor& new_param) {
            // For remove, we select only the surviving Gaussians' optimizer state
            size_t old_size = state.size;
            state.exp_avg = state.exp_avg.index_select(0, sampled_idxs);
            state.exp_avg_sq = state.exp_avg_sq.index_select(0, sampled_idxs);
            // CRITICAL: Update size to match new parameter count
            size_t new_size = sampled_idxs.shape()[0];
            state.size = new_size;
            LOG_DEBUG("  remove() size update: {} -> {} (removed {})", old_size, new_size, old_size - new_size);
        };

        update_param_with_optimizer(param_fn, optimizer_fn, _optimizer, *_splat_data);
    }

    void DefaultStrategy::prune_gs(int iter) {
        // Check for low opacity
        lfs::core::Tensor is_prune = _splat_data->get_opacity() < _params->prune_opacity;

        auto rotation_raw = _splat_data->rotation_raw();
        is_prune = is_prune.logical_or((rotation_raw * rotation_raw).sum(-1, false) < 1e-8f);

        // Check for too large Gaussians
        if (iter > _params->reset_every) {
            const lfs::core::Tensor max_values = _splat_data->get_scaling().max(-1, false);
            lfs::core::Tensor is_too_big = max_values > _params->prune_scale3d * _splat_data->get_scene_scale();
            is_prune = is_prune.logical_or(is_too_big);
        }

        const auto num_prunes = static_cast<int64_t>(is_prune.sum_scalar());
        if (num_prunes > 0) {
            remove(is_prune);
        }
    }

    void DefaultStrategy::reset_opacity() {
        const auto threshold = 2.0f * _params->prune_opacity;

        const auto param_fn = [&threshold](const int i, const lfs::core::Tensor& param) {
            if (i == 5) {
                // For opacity parameter, clamp to logit(threshold)
                const float logit_threshold = std::log(threshold / (1.0f - threshold));
                return param.clamp_max(logit_threshold);
            }
            LOG_ERROR("Invalid parameter index for reset_opacity: {}", i);
            return param;
        };

        const auto optimizer_fn = [](AdamParamState& state, const lfs::core::Tensor& new_param) {
            // Reset optimizer state for opacity to zeros
            state.exp_avg = lfs::core::Tensor::zeros_like(state.exp_avg);
            state.exp_avg_sq = lfs::core::Tensor::zeros_like(state.exp_avg_sq);
        };

        update_param_with_optimizer(param_fn, optimizer_fn, _optimizer, *_splat_data, {5});
    }

    void DefaultStrategy::post_backward(int iter, RenderOutput& render_output) {
        // Increment SH degree every 1000 iterations
        if (iter % _params->sh_degree_interval == 0) {
            _splat_data->increment_sh_degree();
        }

        if (iter == _params->stop_refine) {
            // Reset densification info at the end of refinement. Saves memory and processing time.
            _splat_data->_densification_info = lfs::core::Tensor::empty({0});
        }

        if (iter >= _params->stop_refine) {
            return;
        }

        if (is_refining(iter)) {
            grow_gs(iter);
            prune_gs(iter);

            _splat_data->_densification_info = lfs::core::Tensor::zeros(
                {2, static_cast<size_t>(_splat_data->size())},
                _splat_data->means().device());
        }

        if (iter % _params->reset_every == 0 && iter > 0) {
            reset_opacity();
        }
    }

    void DefaultStrategy::step(int iter) {
        if (iter < _params->iterations) {
            _optimizer->step(iter);
            _optimizer->zero_grad(iter);
            _scheduler->step();
        }
    }

    // ===== Serialization =====

    namespace {
        constexpr uint32_t DEFAULT_MAGIC = 0x4C464446;  // "LFDF"
        constexpr uint32_t DEFAULT_VERSION = 1;
    }

    void DefaultStrategy::serialize(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&DEFAULT_MAGIC), sizeof(DEFAULT_MAGIC));
        os.write(reinterpret_cast<const char*>(&DEFAULT_VERSION), sizeof(DEFAULT_VERSION));

        // Serialize optimizer state
        if (_optimizer) {
            uint8_t has_optimizer = 1;
            os.write(reinterpret_cast<const char*>(&has_optimizer), sizeof(has_optimizer));
            _optimizer->serialize(os);
        } else {
            uint8_t has_optimizer = 0;
            os.write(reinterpret_cast<const char*>(&has_optimizer), sizeof(has_optimizer));
        }

        // Serialize scheduler state
        if (_scheduler) {
            uint8_t has_scheduler = 1;
            os.write(reinterpret_cast<const char*>(&has_scheduler), sizeof(has_scheduler));
            _scheduler->serialize(os);
        } else {
            uint8_t has_scheduler = 0;
            os.write(reinterpret_cast<const char*>(&has_scheduler), sizeof(has_scheduler));
        }

        LOG_DEBUG("Serialized DefaultStrategy");
    }

    void DefaultStrategy::deserialize(std::istream& is) {
        uint32_t magic, version;
        is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        is.read(reinterpret_cast<char*>(&version), sizeof(version));

        if (magic != DEFAULT_MAGIC) {
            throw std::runtime_error("Invalid DefaultStrategy checkpoint: wrong magic");
        }
        if (version != DEFAULT_VERSION) {
            throw std::runtime_error("Unsupported DefaultStrategy checkpoint version: " + std::to_string(version));
        }

        // Deserialize optimizer state
        uint8_t has_optimizer;
        is.read(reinterpret_cast<char*>(&has_optimizer), sizeof(has_optimizer));
        if (has_optimizer && _optimizer) {
            _optimizer->deserialize(is);
        }

        // Deserialize scheduler state
        uint8_t has_scheduler;
        is.read(reinterpret_cast<char*>(&has_scheduler), sizeof(has_scheduler));
        if (has_scheduler && _scheduler) {
            _scheduler->deserialize(is);
        }

        LOG_DEBUG("Deserialized DefaultStrategy");
    }

    void DefaultStrategy::reserve_optimizer_capacity(size_t capacity) {
        if (_optimizer) {
            _optimizer->reserve_capacity(capacity);
            LOG_INFO("Reserved optimizer capacity for {} Gaussians", capacity);
        }
    }

} // namespace lfs::training
