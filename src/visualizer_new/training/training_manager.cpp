/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "training/training_manager.hpp"
#include "core_new/events.hpp"
#include "core_new/logger.hpp"
#include "scene/scene.hpp"
#include "training_new/training_setup.hpp"
#include <cstring>
#include <cuda_runtime.h>
#include <stdexcept>

namespace lfs::vis {

    using namespace lfs::core::events;

    TrainerManager::TrainerManager() {
        setupEventHandlers();
        LOG_DEBUG("TrainerManager created");
    }

    TrainerManager::~TrainerManager() {
        // Ensure training is stopped before destruction
        if (training_thread_ && training_thread_->joinable()) {
            LOG_INFO("Stopping training thread during destruction...");
            stopTraining();
            waitForCompletion();
        }
    }

    void TrainerManager::setTrainer(std::unique_ptr<lfs::training::Trainer> trainer) {
        LOG_TIMER_TRACE("TrainerManager::setTrainer");

        // Clear any existing trainer first
        clearTrainer();

        if (trainer) {
            LOG_DEBUG("Setting new trainer");
            trainer_ = std::move(trainer);
            trainer_->setProject(project_);
            setState(State::Ready);

            // Trainer is ready
            lfs::core::events::internal::TrainerReady{}.emit();
            LOG_INFO("Trainer ready for training");
        }
    }

    void TrainerManager::setTrainerFromCheckpoint(std::unique_ptr<lfs::training::Trainer> trainer, int checkpoint_iteration) {
        LOG_TIMER_TRACE("TrainerManager::setTrainerFromCheckpoint");

        // Clear any existing trainer first
        clearTrainer();

        if (trainer) {
            LOG_DEBUG("Setting trainer from checkpoint (iteration {})", checkpoint_iteration);
            trainer_ = std::move(trainer);
            trainer_->setProject(project_);

            // Set to Ready state - user will click "Resume" or "Start" to begin training
            // The iteration is already set in the trainer from load_checkpoint
            setState(State::Ready);

            // Trainer is ready
            lfs::core::events::internal::TrainerReady{}.emit();
            LOG_INFO("Trainer ready from checkpoint at iteration {} (state: Ready)", checkpoint_iteration);
        }
    }

    bool TrainerManager::hasTrainer() const {
        return trainer_ != nullptr;
    }

    void TrainerManager::clearTrainer() {
        LOG_DEBUG("Clearing trainer");

        lfs::core::events::cmd::StopTraining{}.emit();
        // Stop any ongoing training first
        if (isTrainingActive()) {
            LOG_INFO("Stopping active training before clearing trainer");
            stopTraining();
            waitForCompletion();
        }

        // Additional safety: ensure thread is properly stopped even if not "active"
        if (training_thread_ && training_thread_->joinable()) {
            LOG_WARN("Force stopping training thread that wasn't marked as active");
            training_thread_->request_stop();

            // Try to wait for completion with a short timeout
            auto timeout = std::chrono::milliseconds(500);
            {
                std::unique_lock<std::mutex> lock(completion_mutex_);
                if (completion_cv_.wait_for(lock, timeout, [this] { return training_complete_; })) {
                    lock.unlock();
                    LOG_DEBUG("Thread completed gracefully, joining...");
                    training_thread_->join();
                } else {
                    lock.unlock();
                    LOG_WARN("Thread didn't respond to stop request within timeout, detaching...");
                    training_thread_->detach();
                }
            }
            training_thread_.reset();
        }

        // Now safe to clear the trainer
        trainer_.reset();
        last_error_.clear();
        setState(State::Idle);

        // Reset loss buffer
        loss_buffer_.clear();
        LOG_INFO("Trainer cleared");
    }

    std::expected<bool, std::string> TrainerManager::initializeTrainerFromProject() {
        if (!trainer_) {
            return std::unexpected("No trainer available");
        }

        if (!project_) {
            return std::unexpected("No project available");
        }

        // Create training parameters from project
        lfs::core::param::TrainingParameters params;

        // Convert lfs::project::DataSetInfo to lfs::core::param::DatasetConfig
        // DataSetInfo inherits from DatasetConfig, so we can slice-copy the base
        const auto& updated_dataset = project_->getProjectData().data_set_info;
        params.dataset = static_cast<lfs::core::param::DatasetConfig>(updated_dataset);

        // Project now returns lfs::core::param::OptimizationParameters directly (no conversion needed)
        params.optimization = project_->getOptimizationParams();

        // Initialize trainer
        auto init_result = trainer_->initialize(params);
        if (!init_result) {
            return std::unexpected(init_result.error());
        }

        return true;
    }

    bool TrainerManager::startTraining() {
        LOG_TIMER("TrainerManager::startTraining");

        if (!canStart()) {
            LOG_WARN("Cannot start training in current state: {}", static_cast<int>(state_.load()));
            return false;
        }

        if (!trainer_) {
            LOG_ERROR("Cannot start training - no trainer available");
            return false;
        }

        // Skip initialization if trainer is already initialized (e.g., resuming from checkpoint)
        if (trainer_->isInitialized()) {
            LOG_INFO("Trainer already initialized (resuming from iteration {}), skipping reinitialization",
                     trainer_->get_current_iteration());
        } else {
            // Initialize training model from PointCloud if needed
            // This converts PointCloud node to SplatData, applying any CropBox filtering
            if (scene_ && project_) {
                lfs::core::param::TrainingParameters params;
                const auto& updated_dataset = project_->getProjectData().data_set_info;
                params.dataset = static_cast<lfs::core::param::DatasetConfig>(updated_dataset);
                params.optimization = project_->getOptimizationParams();

                auto model_init_result = lfs::training::initializeTrainingModel(params, *scene_);
                if (!model_init_result) {
                    LOG_ERROR("Failed to initialize training model: {}", model_init_result.error());
                    last_error_ = model_init_result.error();
                    setState(State::Error);
                    return false;
                }
            }

            // Reinitialize trainer to pick up any parameter changes from the project
            LOG_INFO("Initializing trainer with current project parameters");
            auto init_result = initializeTrainerFromProject();
            if (!init_result) {
                LOG_ERROR("Failed to initialize trainer: {}", init_result.error());
                last_error_ = init_result.error();
                setState(State::Error);
                return false;
            }
        }

        // Reset completion state
        {
            std::lock_guard<std::mutex> lock(completion_mutex_);
            training_complete_ = false;
        }

        setState(State::Running);

        // Emit training started event
        state::TrainingStarted{
            .total_iterations = getTotalIterations()}
            .emit();

        // Start training thread
        training_thread_ = std::make_unique<std::jthread>(
            [this](std::stop_token stop_token) {
                trainingThreadFunc(stop_token);
            });

        LOG_INFO("Training started - {} iterations planned", getTotalIterations());
        return true;
    }

    void TrainerManager::pauseTraining() {
        if (!canPause()) {
            LOG_TRACE("Cannot pause training in current state: {}", static_cast<int>(state_.load()));
            return;
        }

        if (trainer_) {
            trainer_->request_pause();
            setState(State::Paused);

            state::TrainingPaused{
                .iteration = getCurrentIteration()}
                .emit();

            LOG_INFO("Training paused at iteration {} (state: Paused)", getCurrentIteration());
        }
    }

    void TrainerManager::resumeTraining() {
        if (!canResume()) {
            LOG_TRACE("Cannot resume training in current state: {}", static_cast<int>(state_.load()));
            return;
        }

        if (trainer_) {
            trainer_->request_resume();
            setState(State::Running);

            state::TrainingResumed{
                .iteration = getCurrentIteration()}
                .emit();

            LOG_INFO("Training resumed from iteration {} (state: Running)", getCurrentIteration());
        }
    }

    void TrainerManager::pauseTrainingTemporary() {
        // Temporary pause for camera movement - doesn't change state
        if (state_ != State::Running) {
            return;
        }

        if (trainer_) {
            trainer_->request_pause();
            LOG_TRACE("Training temporarily paused at iteration {} (state remains Running)", getCurrentIteration());
        }
    }

    void TrainerManager::resumeTrainingTemporary() {
        // Resume from temporary pause - only if still in Running state
        if (state_ != State::Running) {
            return;
        }

        if (trainer_) {
            trainer_->request_resume();
            LOG_TRACE("Training resumed from temporary pause at iteration {} (state remains Running)", getCurrentIteration());
        }
    }

    void TrainerManager::stopTraining() {
        if (!isTrainingActive()) {
            LOG_TRACE("Training not active, nothing to stop");
            return;
        }

        LOG_DEBUG("Requesting training stop");
        setState(State::Stopping);

        if (trainer_) {
            trainer_->request_stop();
        }

        if (training_thread_ && training_thread_->joinable()) {
            LOG_DEBUG("Requesting training thread to stop...");
            training_thread_->request_stop();
        }

        state::TrainingStopped{
            .iteration = getCurrentIteration(),
            .user_requested = true}
            .emit();

        LOG_INFO("Training stop requested at iteration {}", getCurrentIteration());
    }

    void TrainerManager::requestSaveCheckpoint() {
        if (trainer_ && isTrainingActive()) {
            trainer_->request_save();
            LOG_INFO("Checkpoint save requested at iteration {}", getCurrentIteration());
        } else {
            LOG_WARN("Cannot save checkpoint - training not active");
        }
    }

    bool TrainerManager::resetTraining() {
        LOG_INFO("Resetting training to initial state");

        if (!trainer_) {
            LOG_WARN("No trainer to reset");
            return false;
        }

        // Stop if active
        if (isTrainingActive()) {
            stopTraining();
            waitForCompletion();
        }

        if (trainer_->isInitialized()) {
            LOG_DEBUG("Clearing GPU memory from previous training");

            // Save params before destroying
            auto params = trainer_->getParams();

            // Destroy the trainer to release all tensors
            trainer_.reset();

            // Synchronize to ensure all GPU operations are complete
            cudaDeviceSynchronize();

            LOG_DEBUG("GPU memory released");

            // Recreate trainer from Scene
            if (!scene_) {
                LOG_ERROR("Cannot reset training: no scene set");
                setState(State::Error);
                return false;
            }
            trainer_ = std::make_unique<lfs::training::Trainer>(*scene_);
            trainer_->setProject(project_);
        }

        // Clear loss buffer
        loss_buffer_.clear();

        // Set to Ready state
        setState(State::Ready);

        LOG_INFO("Training reset complete - GPU memory freed, ready to start with current parameters");
        return true;
    }

    void TrainerManager::waitForCompletion() {
        if (!training_thread_ || !training_thread_->joinable()) {
            return;
        }

        LOG_DEBUG("Waiting for training thread to complete...");

        std::unique_lock<std::mutex> lock(completion_mutex_);
        completion_cv_.wait(lock, [this] { return training_complete_; });

        training_thread_->join();
        training_thread_.reset();

        LOG_DEBUG("Training thread joined successfully");
    }

    int TrainerManager::getCurrentIteration() const {
        return trainer_ ? trainer_->get_current_iteration() : 0;
    }

    float TrainerManager::getCurrentLoss() const {
        return trainer_ ? trainer_->get_current_loss() : 0.0f;
    }

    int TrainerManager::getTotalIterations() const {
        if (!trainer_)
            return 0;
        return trainer_->getParams().optimization.iterations;
    }

    int TrainerManager::getNumSplats() const {
        if (!trainer_)
            return 0;
        // Strategy may not be created yet if using Scene-based constructor
        // In that case, try to get size from scene
        if (scene_) {
            const auto* model = scene_->getTrainingModel();
            if (model) {
                return static_cast<int>(model->size());
            }
        }
        // Fall back to strategy if trainer is initialized
        if (trainer_->isInitialized()) {
            return static_cast<int>(trainer_->get_strategy().get_model().size());
        }
        return 0;
    }

    void TrainerManager::updateLoss(float loss) {
        std::lock_guard<std::mutex> lock(loss_buffer_mutex_);
        loss_buffer_.push_back(loss);
        while (loss_buffer_.size() > static_cast<size_t>(max_loss_points_)) {
            loss_buffer_.pop_front();
        }
        LOG_TRACE("Loss updated: {:.6f} (buffer size: {})", loss, loss_buffer_.size());
    }

    std::deque<float> TrainerManager::getLossBuffer() const {
        std::lock_guard<std::mutex> lock(loss_buffer_mutex_);
        return loss_buffer_;
    }

    void TrainerManager::trainingThreadFunc(std::stop_token stop_token) {
        LOG_INFO("Training thread started");
        LOG_TIMER("Training execution");

        try {
            LOG_DEBUG("Starting trainer->train() with stop token");
            auto train_result = trainer_->train(stop_token);

            if (!train_result) {
                LOG_ERROR("Training failed: {}", train_result.error());
                handleTrainingComplete(false, train_result.error());
            } else {
                LOG_INFO("Training completed successfully");
                handleTrainingComplete(true);
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Exception in training thread: {}", e.what());
            handleTrainingComplete(false, std::format("Exception in training: {}", e.what()));
        } catch (...) {
            LOG_CRITICAL("Unknown exception in training thread");
            handleTrainingComplete(false, "Unknown exception in training");
        }

        LOG_INFO("Training thread finished");
    }

    void TrainerManager::setState(State new_state) {
        std::lock_guard<std::mutex> lock(state_mutex_);

        State old_state = state_.load();
        state_ = new_state;

        const char* state_str = "";
        switch (new_state) {
        case State::Idle: state_str = "Idle"; break;
        case State::Ready: state_str = "Ready"; break;
        case State::Running: state_str = "Running"; break;
        case State::Paused: state_str = "Paused"; break;
        case State::Stopping: state_str = "Stopping"; break;
        case State::Completed: state_str = "Completed"; break;
        case State::Error: state_str = "Error"; break;
        }

        LOG_DEBUG("Training state changed from {} to {}",
                  static_cast<int>(old_state), state_str);
    }

    void TrainerManager::handleTrainingComplete(bool success, const std::string& error) {
        if (!error.empty()) {
            last_error_ = error;
            LOG_ERROR("Training error: {}", error);
        }

        setState(success ? State::Completed : State::Error);

        int final_iteration = getCurrentIteration();
        float final_loss = getCurrentLoss();

        LOG_INFO("Training finished - Success: {}, Final iteration: {}, Final loss: {:.6f}",
                 success, final_iteration, final_loss);

        state::TrainingCompleted{
            .iteration = final_iteration,
            .final_loss = final_loss,
            .success = success,
            .error = error.empty() ? std::nullopt : std::optional(error)}
            .emit();

        // Notify completion
        {
            std::lock_guard<std::mutex> lock(completion_mutex_);
            training_complete_ = true;
        }
        completion_cv_.notify_all();
    }

    void TrainerManager::setupEventHandlers() {
        using namespace lfs::core::events;

        // Listen for training progress events - only update loss buffer
        state::TrainingProgress::when([this](const auto& event) {
            updateLoss(event.loss);
        });
    }

    std::shared_ptr<const lfs::core::Camera> TrainerManager::getCamById(int camId) const {
        // Get camera from Scene (Scene owns all training data)
        if (scene_) {
            return scene_->getCameraByUid(camId);
        }
        LOG_ERROR("getCamById called but scene is not set");
        return nullptr;
    }

    std::vector<std::shared_ptr<const lfs::core::Camera>> TrainerManager::getCamList() const {
        // Get cameras from Scene (Scene owns all training data)
        if (scene_) {
            return scene_->getAllCameras();
        }
        LOG_ERROR("getCamList called but scene is not set");
        return {};
    }

    void TrainerManager::setProject(std::shared_ptr<lfs::project::Project> project) {
        project_ = project;
        if (trainer_) {
            trainer_->setProject(project);
        }
    }

} // namespace lfs::vis