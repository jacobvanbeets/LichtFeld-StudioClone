/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <string>

#ifdef _WIN32
#include <BaseTsd.h>
#include <windows.h>
using ssize_t = SSIZE_T;
#else
#include <sys/types.h>
#endif

namespace lfs::vis::terminal {

    class PtyProcess {
    public:
        static constexpr int DEFAULT_COLS = 80;
        static constexpr int DEFAULT_ROWS = 24;

        PtyProcess() = default;
        ~PtyProcess();

        PtyProcess(const PtyProcess&) = delete;
        PtyProcess& operator=(const PtyProcess&) = delete;
        PtyProcess(PtyProcess&& other) noexcept;
        PtyProcess& operator=(PtyProcess&& other) noexcept;

        bool spawn(const std::string& shell = "", int cols = DEFAULT_COLS, int rows = DEFAULT_ROWS);
        void close();
        [[nodiscard]] bool is_running() const;

        [[nodiscard]] ssize_t read(char* buf, size_t len);
        [[nodiscard]] ssize_t write(const char* buf, size_t len);

        void resize(int cols, int rows);
        void interrupt();

        [[nodiscard]] int fd() const;

    private:
        void cleanup();

#ifdef _WIN32
        HPCON hpc_ = nullptr;
        HANDLE pipe_in_ = INVALID_HANDLE_VALUE;
        HANDLE pipe_out_ = INVALID_HANDLE_VALUE;
        HANDLE process_ = INVALID_HANDLE_VALUE;
        HANDLE thread_ = INVALID_HANDLE_VALUE;
#else
        int master_fd_ = -1;
        pid_t child_pid_ = -1;
#endif
    };

} // namespace lfs::vis::terminal
