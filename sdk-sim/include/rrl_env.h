/*───────────────────────────────────────────────────────────
 *  rrl_env.h  —  Public C ABI for RemoteRL Sim-SDK
 *
 *  This header is **stable**: once published, we only add symbols
 *  (never remove or change existing ones) to preserve binary
 *  compatibility across SDK updates.
 *
 *  Five functions (poll / stats / load_policy / last_error /
 *  last_error_msg) are intentionally left for engine integrators to
 *  override — either by providing stronger symbols at link-time or
 *  via rrl_register_backend() at run-time.
 *───────────────────────────────────────────────────────────*/
#ifndef RRL_ENV_H
#define RRL_ENV_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>  /* size_t */

/*────────────────── Error codes ──────────────────────────*/
enum {
    RRL_SUCCESS              =  0,
    RRL_ERR_INVALID_HANDLE   = -1,
    RRL_ERR_INVALID_ARGUMENT = -2,
    RRL_ERR_UNSUPPORTED      = -3,
    RRL_ERR_NO_BACKEND       = -4,
};

/*────────────────── Opaque handle ────────────────────────*/
typedef struct RRLHandleImpl *RRLHandle; /* defined inside closed core */

/*────────────────── Space descriptor ─────────────────────*/
typedef struct {
    int shape[8];    /* tensor dimensions (up to 8-D)          */
    int ndim;        /* number of valid dims in shape[]         */
    int dtype;       /* RRL_DTYPE_*  (enum defined elsewhere)   */
} RRL_SpaceDesc;

/*────────────────── Stats snapshot ───────────────────────*/
typedef struct {
    double        fps;        /* sim frames/second               */
    double        latency_ms; /* mean round-trip latency (ms)    */
    unsigned long steps;      /* total environment steps taken   */
} RRL_Stats;

/*────────────────── Core metadata API ────────────────────*/
int rrl_action_space(RRLHandle handle, RRL_SpaceDesc *out_space);
int rrl_observation_space(RRLHandle handle, RRL_SpaceDesc *out_space);

/* Legacy aliases (deprecated) */
#define rrl_get_action_space       rrl_action_space
#define rrl_get_observation_space rrl_observation_space

/*────────────────── User-pluggable backend ───────────────*/
typedef struct {
    int (*poll)       (RRLHandle);
    int (*get_stats)  (RRLHandle, RRL_Stats *);
    int (*load_policy)(RRLHandle, const void *bytes, size_t len);
} RRL_BackendHooks;

/* Register custom function table (pass NULL to restore stubs) */
int rrl_register_backend(const RRL_BackendHooks *hooks);

/*────────────────── Five overridable exports ─────────────*/
int         rrl_poll            (RRLHandle handle);                                /* 0/1 */
int         rrl_get_stats       (RRLHandle handle, RRL_Stats *out_stats);          /* RRL_SUCCESS / err */
int         rrl_load_policy     (RRLHandle handle, const void *bytes, size_t len); /* RRL_SUCCESS / err */
int         rrl_last_error      (void);                                            /* returns last err code */
const char *rrl_last_error_msg  (void);                                            /* human-readable */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RRL_ENV_H */
