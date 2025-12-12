/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "istrategy.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "optimizer/scheduler.hpp"
#include <memory>

namespace lfs::training {
    // Forward declarations
    struct RenderOutput;

    /// Default densification-based optimization strategy. SplatData owned by Scene.
    class DefaultStrategy : public IStrategy {
    public:
        DefaultStrategy() = delete;
        /// SplatData must be owned by Scene
        explicit DefaultStrategy(lfs::core::SplatData& splat_data);

        // Prevent copy/move
        DefaultStrategy(const DefaultStrategy&) = delete;
        DefaultStrategy& operator=(const DefaultStrategy&) = delete;
        DefaultStrategy(DefaultStrategy&&) = delete;
        DefaultStrategy& operator=(DefaultStrategy&&) = delete;

        // IStrategy interface implementation
        void initialize(const lfs::core::param::OptimizationParameters& optimParams) override;

        void post_backward(int iter, RenderOutput& render_output) override;

        void step(int iter) override;

        bool is_refining(int iter) const override;

        lfs::core::SplatData& get_model() override { return *_splat_data; }
        const lfs::core::SplatData& get_model() const override { return *_splat_data; }

        void remove_gaussians(const lfs::core::Tensor& mask) override;

        // IStrategy interface - optimizer access
        AdamOptimizer& get_optimizer() override { return *_optimizer; }
        const AdamOptimizer& get_optimizer() const override { return *_optimizer; }
        ExponentialLR* get_scheduler() { return _scheduler.get(); }
        const ExponentialLR* get_scheduler() const { return _scheduler.get(); }

        // Serialization for checkpoints
        void serialize(std::ostream& os) const override;
        void deserialize(std::istream& is) override;
        const char* strategy_type() const override { return "default"; }

        // Reserve optimizer capacity for future growth (e.g., after checkpoint load)
        void reserve_optimizer_capacity(size_t capacity) override;

    private:
        // Helper functions
        void duplicate(const lfs::core::Tensor& is_duplicated);

        void split(const lfs::core::Tensor& is_split);

        void grow_gs(int iter);

        void remove(const lfs::core::Tensor& is_prune);

        void prune_gs(int iter);

        void reset_opacity();

        // Member variables
        std::unique_ptr<AdamOptimizer> _optimizer;
        std::unique_ptr<ExponentialLR> _scheduler;
        lfs::core::SplatData* _splat_data = nullptr;  // Scene-owned
        std::unique_ptr<const lfs::core::param::OptimizationParameters> _params;
    };
} // namespace lfs::training
