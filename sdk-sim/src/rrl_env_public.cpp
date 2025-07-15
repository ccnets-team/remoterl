
//─────────────────────────────────────────────────────────────
//  rrl_env_public.cpp  —  Reference implementation (open part)
//  for the five user‑pluggable RemoteRL Sim‑SDK C functions.
//
//  • Can be compiled as C++17 (or later) and linked together
//    with the closed‑source core library.
//  • All exported symbols keep **C linkage** so that the final
//    binary ABI matches rrl_env.h exactly.
//  • Default bodies are marked **weak** so that engine teams can
//    override them by simply providing stronger definitions.
//  • Alternatively, call `rrl_register_backend()` at runtime to
//    inject function pointers (hot‑swap).
//─────────────────────────────────────────────────────────────
#include "rrl_env.h"

#include <cstring>   // std::memset, std::strncpy
#include <string>
#include <mutex>

//─────────────────────────────────────────────────────────────
//  Portable weak‑symbol attribute
//─────────────────────────────────────────────────────────────
#if defined(__GNUC__) || defined(__clang__)
#  define RRL_WEAK __attribute__((weak))
#elif defined(_MSC_VER)
#  define RRL_WEAK __declspec(selectany)
#else
#  define RRL_WEAK  /* weak not available → link‑time override only */
#endif

//─────────────────────────────────────────────────────────────
//  Internal helpers / state
//─────────────────────────────────────────────────────────────
namespace {

// Simple thread‑safe error store (per‑thread would be better, but
// we keep global + mutex here for clarity)
static std::mutex       g_err_mtx;
static int              g_last_err   = RRL_SUCCESS;
static char             g_last_msg[256] = "";

void set_error(int code, const char* msg)
{
    std::lock_guard<std::mutex> lk(g_err_mtx);
    g_last_err = code;
    if (msg) {
        std::strncpy(g_last_msg, msg, sizeof(g_last_msg)-1);
        g_last_msg[sizeof(g_last_msg)-1] = '\0';
    } else {
        g_last_msg[0] = '\0';
    }
}

// Runtime‑switchable backend table (may be nullptr)
static const RRL_BackendHooks* g_backend = nullptr;

// Default stub helpers (weak) — engine can replace by defining its
// own versions with the same signature *without* weak attribute.
RRL_WEAK int stub_poll(RRLHandle /*h*/)                            { return 0; }
RRL_WEAK int stub_get_stats(RRLHandle /*h*/, RRL_Stats* s)         { if (s) std::memset(s,0,sizeof(*s)); return RRL_ERR_UNSUPPORTED; }
RRL_WEAK int stub_load_policy(RRLHandle /*h*/, const void*, size_t){ return RRL_ERR_UNSUPPORTED; }

} // namespace (anonymous)

//─────────────────────────────────────────────────────────────
//  Public: register backend (C linkage)
//─────────────────────────────────────────────────────────────
extern "C" int rrl_register_backend(const RRL_BackendHooks* hooks)
{
    g_backend = hooks;  // can be nullptr to reset to stubs
    return RRL_SUCCESS;
}

//─────────────────────────────────────────────────────────────
//  Public exported C functions (overridable)
//─────────────────────────────────────────────────────────────

extern "C" {

int RRL_WEAK rrl_poll(RRLHandle handle)
{
    if (!handle) {
        set_error(RRL_ERR_INVALID_HANDLE, "rrl_poll: null handle");
        return 0;
    }
    int rc = g_backend && g_backend->poll ? g_backend->poll(handle)
                                          : stub_poll(handle);
    if (rc < 0) {
        set_error(rc, "rrl_poll: backend error");
        return 0;
    }
    return rc; // 0 or 1
}

int RRL_WEAK rrl_get_stats(RRLHandle handle, RRL_Stats* out_stats)
{
    if (!handle || !out_stats) {
        set_error(RRL_ERR_INVALID_ARGUMENT, "rrl_get_stats: null arg");
        return RRL_ERR_INVALID_ARGUMENT;
    }
    int rc = g_backend && g_backend->get_stats ?
             g_backend->get_stats(handle, out_stats) :
             stub_get_stats(handle, out_stats);
    if (rc != RRL_SUCCESS) {
        set_error(rc, "rrl_get_stats: backend error");
    }
    return rc;
}

int RRL_WEAK rrl_load_policy(RRLHandle handle, const void* bytes, size_t len)
{
    if (!handle) {
        set_error(RRL_ERR_INVALID_HANDLE, "rrl_load_policy: null handle");
        return RRL_ERR_INVALID_HANDLE;
    }
    if (!bytes || len == 0) {
        set_error(RRL_ERR_INVALID_ARGUMENT, "rrl_load_policy: empty blob");
        return RRL_ERR_INVALID_ARGUMENT;
    }
    int rc = g_backend && g_backend->load_policy ?
             g_backend->load_policy(handle, bytes, len) :
             stub_load_policy(handle, bytes, len);
    if (rc != RRL_SUCCESS) {
        set_error(rc, "rrl_load_policy: backend error");
    }
    return rc;
}

int rrl_last_error(void)
{
    std::lock_guard<std::mutex> lk(g_err_mtx);
    return g_last_err;
}

const char* rrl_last_error_msg(void)
{
    std::lock_guard<std::mutex> lk(g_err_mtx);
    return g_last_msg;
}

} // extern "C"
