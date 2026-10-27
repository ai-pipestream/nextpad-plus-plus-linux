/*
 * CancellableSpawn
 *
 * Starts a child process and waits for it. Another thread may cancel()
 * the child, or retire() the spawner so later run() calls do not start
 * anything. Editor shutdown uses retire() so a model download cannot
 * hold the process open for curl's full timeout.
 */
#ifndef NPP_CANCELLABLE_SPAWN_H
#define NPP_CANCELLABLE_SPAWN_H

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

class CancellableSpawn {
public:
    struct Result {
        bool started = false;
        bool cancelled = false;
        bool exited = false;
        int exit_code = -1;
        std::string stderr_text;
        std::string spawn_error;
    };

    Result run(const std::vector<std::string> &args);
    void cancel();
    void retire();
    bool retired() const { return retired_.load(); }

private:
    std::mutex mu_;
    std::atomic<bool> abort_{false};
    std::atomic<bool> retired_{false};
    int pid_ = 0;
};

#endif
