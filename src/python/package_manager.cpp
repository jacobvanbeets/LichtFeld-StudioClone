/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "package_manager.hpp"

#include <core/cuda_version.hpp>
#include <core/logger.hpp>

#include <cstdio>
#include <regex>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace lfs::python {

    namespace {

#ifdef _WIN32
        constexpr const char* UV_BINARY = "uv.exe";
        constexpr size_t MAX_PATH_LEN = MAX_PATH;
#else
        constexpr const char* UV_BINARY = "uv";
        constexpr size_t MAX_PATH_LEN = 4096;
#endif
        constexpr const char* PYTORCH_INDEX = "https://download.pytorch.org/whl/";

        std::filesystem::path get_executable_dir() {
#ifdef _WIN32
            wchar_t path[MAX_PATH_LEN];
            GetModuleFileNameW(nullptr, path, MAX_PATH_LEN);
            return std::filesystem::path(path).parent_path();
#else
            char path[MAX_PATH_LEN];
            const ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
            if (len != -1) {
                path[len] = '\0';
                return std::filesystem::path(path).parent_path();
            }
            return std::filesystem::current_path();
#endif
        }

        std::pair<int, std::string> execute_command(const std::string& cmd) {
            std::string output;
            int exit_code = -1;

#ifdef _WIN32
            SECURITY_ATTRIBUTES sa;
            sa.nLength = sizeof(SECURITY_ATTRIBUTES);
            sa.bInheritHandle = TRUE;
            sa.lpSecurityDescriptor = nullptr;

            HANDLE hReadPipe, hWritePipe;
            if (!CreatePipe(&hReadPipe, &hWritePipe, &sa, 0)) {
                return {-1, "Failed to create pipe"};
            }

            SetHandleInformation(hReadPipe, HANDLE_FLAG_INHERIT, 0);

            STARTUPINFOW si = {};
            si.cb = sizeof(si);
            si.hStdOutput = hWritePipe;
            si.hStdError = hWritePipe;
            si.dwFlags |= STARTF_USESTDHANDLES;

            PROCESS_INFORMATION pi = {};

            std::wstring wcmd(cmd.begin(), cmd.end());
            if (!CreateProcessW(nullptr, wcmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) {
                CloseHandle(hReadPipe);
                CloseHandle(hWritePipe);
                return {-1, "Failed to create process"};
            }

            CloseHandle(hWritePipe);

            char buffer[4096];
            DWORD bytesRead;
            while (ReadFile(hReadPipe, buffer, sizeof(buffer) - 1, &bytesRead, nullptr) && bytesRead > 0) {
                buffer[bytesRead] = '\0';
                output += buffer;
            }

            WaitForSingleObject(pi.hProcess, INFINITE);

            DWORD exitCodeDword;
            GetExitCodeProcess(pi.hProcess, &exitCodeDword);
            exit_code = static_cast<int>(exitCodeDword);

            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);
            CloseHandle(hReadPipe);
#else
            FILE* pipe = popen((cmd + " 2>&1").c_str(), "r");
            if (!pipe) {
                return {-1, "Failed to execute command"};
            }

            char buffer[4096];
            while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                output += buffer;
            }

            exit_code = pclose(pipe);
            if (WIFEXITED(exit_code)) {
                exit_code = WEXITSTATUS(exit_code);
            }
#endif

            return {exit_code, output};
        }

        std::filesystem::path get_lichtfeld_dir() {
#ifdef _WIN32
            const char* const home = std::getenv("USERPROFILE");
#else
            const char* const home = std::getenv("HOME");
#endif
            return std::filesystem::path(home ? home : "/tmp") / ".lichtfeld";
        }

    } // namespace

    PackageManager::PackageManager() : m_venv_dir(get_lichtfeld_dir() / "venv") {}

    PackageManager& PackageManager::instance() {
        static PackageManager inst;
        return inst;
    }

    std::filesystem::path PackageManager::uv_path() const {
        const auto exe_dir = get_executable_dir();

        if (const auto p = exe_dir / "bin" / UV_BINARY; std::filesystem::exists(p))
            return p;
        if (const auto p = exe_dir / UV_BINARY; std::filesystem::exists(p))
            return p;

#ifdef _WIN32
        auto [exit_code, result] = execute_command("where uv");
#else
        auto [exit_code, result] = execute_command("which uv");
#endif
        if (exit_code == 0 && !result.empty()) {
            while (!result.empty() && (result.back() == '\n' || result.back() == '\r'))
                result.pop_back();
            if (std::filesystem::path found(result); std::filesystem::exists(found))
                return found;
        }
        return {};
    }

    bool PackageManager::is_uv_available() const {
        return !uv_path().empty();
    }

    std::filesystem::path PackageManager::venv_dir() const {
        return m_venv_dir;
    }

    std::filesystem::path PackageManager::venv_python() const {
#ifdef _WIN32
        return m_venv_dir / "Scripts" / "python.exe";
#else
        return m_venv_dir / "bin" / "python";
#endif
    }

    bool PackageManager::is_venv_ready() const {
        return m_venv_ready && std::filesystem::exists(venv_python());
    }

    bool PackageManager::ensure_venv() {
        std::lock_guard lock(m_mutex);

        if (m_venv_ready && std::filesystem::exists(venv_python()))
            return true;

        const auto uv = uv_path();
        if (uv.empty()) {
            LOG_ERROR("uv not found");
            return false;
        }

        if (std::filesystem::exists(venv_python())) {
            m_venv_ready = true;
            return true;
        }

        LOG_INFO("Creating venv at {}", m_venv_dir.string());

        std::ostringstream cmd;
        cmd << "\"" << uv.string() << "\" venv \"" << m_venv_dir.string() << "\" --allow-existing";

        const auto [exit_code, output] = execute_command(cmd.str());
        if (exit_code != 0) {
            LOG_ERROR("Failed to create venv: {}", output);
            return false;
        }

        m_venv_ready = true;
        return true;
    }

    std::filesystem::path PackageManager::site_packages_dir() const {
#ifdef _WIN32
        return m_venv_dir / "Lib" / "site-packages";
#else
        const auto lib_dir = m_venv_dir / "lib";
        if (std::filesystem::exists(lib_dir)) {
            for (const auto& entry : std::filesystem::directory_iterator(lib_dir)) {
                if (entry.is_directory()) {
                    const auto name = entry.path().filename().string();
                    if (name.find("python") == 0)
                        return entry.path() / "site-packages";
                }
            }
        }
        return m_venv_dir / "lib" / "python3" / "site-packages";
#endif
    }

    InstallResult PackageManager::execute_uv(const std::vector<std::string>& args) const {
        const auto uv = uv_path();
        if (uv.empty())
            return {.error = "uv not found"};

        std::ostringstream cmd;
        cmd << "\"" << uv.string() << "\"";
        for (const auto& arg : args)
            cmd << " " << arg;

        const auto [exit_code, output] = execute_command(cmd.str());

        InstallResult result;
        result.output = output;
        result.success = (exit_code == 0);
        if (!result.success)
            result.error = output.empty() ? "Exit code " + std::to_string(exit_code) : output;
        return result;
    }

    InstallResult PackageManager::install(const std::string& package) {
        if (!ensure_venv())
            return {.error = "Failed to create venv"};

        std::lock_guard lock(m_mutex);
        LOG_INFO("Installing {}", package);
        return execute_uv({"pip", "install", package, "--python", venv_python().string()});
    }

    InstallResult PackageManager::uninstall(const std::string& package) {
        if (!ensure_venv())
            return {.error = "Failed to initialize venv"};

        std::lock_guard lock(m_mutex);
        LOG_INFO("Uninstalling {}", package);
        return execute_uv({"pip", "uninstall", package, "--python", venv_python().string()});
    }

    InstallResult PackageManager::install_torch(const std::string& cuda_version,
                                                const std::string& torch_version) {
        if (!ensure_venv())
            return {.error = "Failed to create venv"};

        std::string cuda_tag = cuda_version;
        if (cuda_tag == "auto") {
            const auto info = core::check_cuda_version();
            if (info.query_failed) {
                cuda_tag = "cu124";
            } else if (info.major >= 12) {
                cuda_tag = info.minor >= 4 ? "cu124" : "cu121";
            } else if (info.major == 11 && info.minor >= 8) {
                cuda_tag = "cu118";
            } else {
                cuda_tag = "cu118";
            }
            LOG_INFO("CUDA {}.{} -> {}", info.major, info.minor, cuda_tag);
        } else if (cuda_tag == "12.4") {
            cuda_tag = "cu124";
        } else if (cuda_tag == "12.1") {
            cuda_tag = "cu121";
        } else if (cuda_tag == "11.8") {
            cuda_tag = "cu118";
        }

        std::string package = "torch";
        if (!torch_version.empty())
            package += "==" + torch_version;

        const std::string index_url = std::string(PYTORCH_INDEX) + cuda_tag;

        std::lock_guard lock(m_mutex);
        LOG_INFO("Installing {} from {}", package, cuda_tag);

        return execute_uv({"pip", "install", package,
                           "--extra-index-url", index_url,
                           "--python", venv_python().string()});
    }

    std::vector<PackageInfo> PackageManager::list_installed() const {
        std::lock_guard lock(m_mutex);
        std::vector<PackageInfo> packages;

        const auto site_dir = site_packages_dir();
        if (!std::filesystem::exists(site_dir))
            return packages;

        static const std::regex DIST_INFO_PATTERN(R"((.+)-(.+)\.dist-info)");

        for (const auto& entry : std::filesystem::directory_iterator(site_dir)) {
            if (!entry.is_directory())
                continue;

            const auto name = entry.path().filename().string();
            if (name.find(".dist-info") == std::string::npos)
                continue;

            std::smatch match;
            if (std::regex_match(name, match, DIST_INFO_PATTERN))
                packages.push_back({.name = match[1].str(), .version = match[2].str()});
        }
        return packages;
    }

    bool PackageManager::is_installed(const std::string& package) const {
        const auto packages = list_installed();
        auto normalize = [](std::string s) {
            std::replace(s.begin(), s.end(), '-', '_');
            return s;
        };
        const auto normalized = normalize(package);

        for (const auto& pkg : packages) {
            if (pkg.name == package || normalize(pkg.name) == normalized)
                return true;
        }
        return false;
    }

} // namespace lfs::python
