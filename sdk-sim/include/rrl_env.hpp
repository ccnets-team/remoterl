//─────────────────────────────────────────────────────────────
//  rrl_env.hpp  —  C++17 convenience wrapper for RemoteRL C API
//─────────────────────────────────────────────────────────────
//  • Keeps the *stable C ABI* underneath (rrl_env.h)
//  • Adds RAII, std::string, and type-safe enums for C++ users.
//  • Header-only: just #include and link against libremoterl.
//─────────────────────────────────────────────────────────────
#ifndef RRL_ENV_HPP
#define RRL_ENV_HPP

#include <string>
#include <stdexcept>
#include <utility>
#include "rrl_env.h"   // C API generated earlier

namespace rrl {
//──── Strong-typed view wrappers ───────────────────────────//
struct ActionSpace { RRL_SpaceDesc raw; };
struct ObservationSpace { RRL_SpaceDesc raw; };
struct Stats {
    RRL_Stats raw{};
    double      fps()      const noexcept { return raw.fps; }
    double      latency()  const noexcept { return raw.latency_ms; }
    unsigned long steps()  const noexcept { return raw.steps; }
};

//──── RAII handle wrapper ─────────────────────────────────//
class Env {
public:
    explicit Env(RRLHandle h) : h_(h) {
        if (!h_) throw std::invalid_argument("Null RRLHandle passed to Env");
    }
    // Non-copyable (owns raw handle semantics are unclear)
    Env(const Env&)            = delete;
    Env& operator=(const Env&) = delete;
    // Moveable
    Env(Env&& other) noexcept : h_(other.h_) { other.h_ = nullptr; }
    Env& operator=(Env&& o) noexcept { std::swap(h_, o.h_); return *this; }
    ~Env() noexcept { if (h_) rrl_close(h_); }

    ActionSpace action_space() const {
        RRL_SpaceDesc s{};
        if (rrl_action_space(h_, &s) != RRL_SUCCESS)
            throw_error("action_space");
        return {s};
    }

    ObservationSpace observation_space() const {
        RRL_SpaceDesc s{};
        if (rrl_observation_space(h_, &s) != RRL_SUCCESS)
            throw_error("observation_space");
        return {s};
    }

    bool poll() const { return rrl_poll(h_) != 0; }

    Stats stats() const {
        Stats s;
        if (rrl_get_stats(h_, &s.raw) != RRL_SUCCESS)
            throw_error("get_stats");
        return s;
    }

    void load_policy(const void* bytes, size_t len) {
        if (rrl_load_policy(h_, bytes, len) != RRL_SUCCESS)
            throw_error("load_policy");
    }

    // direct raw access if really needed
    RRLHandle raw() const noexcept { return h_; }

private:
    RRLHandle h_{};

    static void throw_error(const char* what) {
        auto code = rrl_last_error();
        const char* msg = rrl_last_error_msg();
        throw std::runtime_error(std::string("RRL ") + what + ": [" + std::to_string(code) + "] " + (msg ? msg : ""));
    }
};

//──── Backend helper (runtime registration) ───────────────//
class Backend {
public:
    using poll_fn  = int(*)(RRLHandle);
    using stats_fn = int(*)(RRLHandle, RRL_Stats*);
    using load_fn  = int(*)(RRLHandle, const void*, size_t);

    constexpr Backend(poll_fn p=nullptr, stats_fn s=nullptr, load_fn l=nullptr)
        : hooks_{p,s,l} {}

    void install() const {
        if (rrl_register_backend(&hooks_) != RRL_SUCCESS) {
            throw std::runtime_error("rrl_register_backend failed");
        }
    }
private:
    RRL_BackendHooks hooks_;
};

} // namespace rrl

#endif /* RRL_ENV_HPP */
