"""
_remote_exception.py - Systematic error handling for RemoteRL SDK.
=====================================================================

Purpose
-------
This module is the **single** place where every exception raised by the
Python SDK is defined. All errors inherit from the common base class
:class:`RemoteRLError` and are organised into seven high-level families:

Seven high-level families
* **AUTH** - authentication & authorisation problems  
* **CFG**  - invalid configuration or missing settings  
* **DEP**  - optional-dependency issues (Gymnasium, Ray, SB3, …)  
* **NET**  - network and transport failures  
* **RUN**  - runtime errors inside an environment or worker  
* **QUO**  - quota or rate-limit violations  
* **MISC** - anything that does not fit the above (fallback)

Typical subclasses you might catch include
:class:`InvalidApiKeyError`, :class:`MissingEnvVariableError`,
and :class:`EnvStepError`.

Why keep every error here?
---------------------

The SDK automatically converts internal errors into these custom
exceptions. Consolidating them in one place improves diagnostics and
user experience by ensuring each exception carries rich runtime context.

For example, if there is an authentication error, the SDK will
raise \:class:`InvalidApiKeyError` with a message like this::

    [RemoteRL] (AUTH-001) Invalid API key – generate a new one
    (details: server reported no key found for this account)

The `_REMOTERL_EXCEPTION_MAP` below includes the code, status, and
headline. The *code* is a short identifier for the error, the *status*
is an HTTP status, and the *headline* is a concise, human-readable
message. The *details* are attached at runtime with the exception
context to customise the message and provide more information about the
error::

    [RemoteRL] (code) headline…
    (details: …)

Furthermore, we plan to improve how we capture exception details and to
classify exceptions more effectively::

    RemoteRLError (catch-all)
        → 7 high-level families (catch all by family)
            → 50+ sub-categories (catch all by sub-category with contextual details)

Version history
---------------
* **v1.1.3** (2025-07-??) - This docstring is published ahead of the
  release for reference; the enhanced exception display (and related
  improvements) will ship with v1.1.3.
"""
from __future__ import annotations

class RemoteRLError(Exception):
    """Convenience root for every *SDK* exception"""

    pass

# ---------------------------------------------------------------------------
# 1) AUTH  - Authentication / Account
# ---------------------------------------------------------------------------
class RemoteRLAuthError(RemoteRLError):
    pass

class NoApiKeyError(RemoteRLAuthError):
    pass

class InvalidApiKeyError(RemoteRLAuthError):
    pass

class ApiKeyExpiredError(RemoteRLAuthError):
    pass

class DataCreditExhaustedError(RemoteRLAuthError):
    pass

class SessionDeniedError(RemoteRLAuthError):
    pass

# ---------------------------------------------------------------------------
# 2) CFG  – Configuration / CLI flags
# ---------------------------------------------------------------------------
class RemoteRLConfigError(RemoteRLError):
    pass

class InvalidFrameworkError(RemoteRLConfigError):
    pass

class EnvNotFoundError(RemoteRLConfigError):
    pass

class InvalidNumWorkersError(RemoteRLConfigError):
    pass

class InvalidNumEnvRunnersError(RemoteRLConfigError):
    pass

class UnknownOptionError(RemoteRLConfigError):
    pass

class DuplicateOptionError(RemoteRLConfigError):
    pass

# ---------------------------------------------------------------------------
# 3) DEP  – Dependencies / Environment
# ---------------------------------------------------------------------------
class RemoteRLDepencyError(RemoteRLError):
    pass

class UnsupportedPythonVersionError(RemoteRLDepencyError):
    pass

class GymnasiumMissingError(RemoteRLDepencyError):
    pass

class SB3MissingError(RemoteRLDepencyError):
    pass

class RLLibMissingError(RemoteRLDepencyError):
    pass

class CleanRLMissingError(RemoteRLDepencyError):
    pass

class PettingZooMissingError(RemoteRLDepencyError):
    pass

class RemoteRLVersionError(RemoteRLDepencyError):
    pass

class RemoteRLVersionMissingErorr(RemoteRLDepencyError):
    pass

class GymnasiumVersionError(RemoteRLDepencyError):
    pass

class SB3VersionError(RemoteRLDepencyError):
    pass

class RLLibVersionError(RemoteRLDepencyError):
    pass

class RLLibraryMissingError(RemoteRLDepencyError):
    pass

class GymnasiumPatchError(RemoteRLDepencyError):
    pass

class StableBaselines3PatchError(RemoteRLDepencyError):
    pass

class RayRLlibPatchError(RemoteRLDepencyError):
    pass

# ---------------------------------------------------------------------------
# 4) NET  – Network / Relay connectivity
# ---------------------------------------------------------------------------
class RemoteRLNetworkError(RemoteRLError):
    pass

class ServerUnreachableError(RemoteRLNetworkError):
    pass

class GlobalRedirectionError(RemoteRLNetworkError):
    pass

class SimulatorSessionOpenError(RemoteRLNetworkError):
    pass

class TrainerSessionOpenError(RemoteRLNetworkError):
    pass

class ServerReachFailedError(RemoteRLNetworkError):
    pass

class ServerListenFailedError(RemoteRLNetworkError):
    pass

class SessionConnectionLostError(RemoteRLNetworkError):
    pass

class RelayConnectionLostError(RemoteRLNetworkError):
    pass

class RelayReconnectionFailError(RemoteRLNetworkError):
    pass

class RemoteRelayTimeoutError(RemoteRLNetworkError):
    pass

class WorkerRelayTimeoutError(RemoteRLNetworkError):
    pass

class EnvRunnerRelayTimeoutError(RemoteRLNetworkError):
    pass

# ---------------------------------------------------------------------------
# 5) RUN  – Runtime / Training loop
# ---------------------------------------------------------------------------
class RemoteRLRuntimeError(RemoteRLError):
    pass

# ‑‑ startup / session -------------------------------------------------------
class SimulatorStartupError(RemoteRLRuntimeError):
    pass

class SimulatorCrashedError(RemoteRLRuntimeError):
    pass

class SimulatorSessionCrashedError(RemoteRLRuntimeError):
    pass

class TrainerStartupError(RemoteRLRuntimeError):
    pass

class TrainerCrashedError(RemoteRLRuntimeError):
    pass

class TrainerSessionCrashedError(RemoteRLRuntimeError):
    pass

# ‑‑ workers / runners -------------------------------------------------------
class WorkerStartupError(RemoteRLRuntimeError):
    pass

class GymWorkerStartupError(RemoteRLRuntimeError):
    pass

class RayWorkerStartupError(RemoteRLRuntimeError):
    pass

class EnvRunnerStartupError(RemoteRLRuntimeError):
    pass

# worker / runner‑side crashes
class WorkerCrashedError(RemoteRLRuntimeError):
    pass

class GymWorkerCrashedError(RemoteRLRuntimeError):
    pass

class RayWorkerCrashedError(RemoteRLRuntimeError):
    pass

class PettingZooWorkerCrashedError(RemoteRLRuntimeError):
    pass

class PettingZooWorkerStartupError(RemoteRLRuntimeError):
    pass

class EnvRunnerCrashedError(RemoteRLRuntimeError):
    pass

# env‑level errors
class EnvRunnerEnvCreationError(RemoteRLRuntimeError):
    pass

class EnvMakeError(RemoteRLRuntimeError):
    pass

class EnvStepError(RemoteRLRuntimeError):
    pass

class EnvResetError(RemoteRLRuntimeError):
    pass

class EnvCloseError(RemoteRLRuntimeError):
    pass

class EnvObservationSpaceError(RemoteRLRuntimeError):
    pass

class EnvActionSpaceError(RemoteRLRuntimeError):
    pass

# shutdown
class ShutdownSetupError(RemoteRLRuntimeError):
    pass

class ShutdownCrashedError(RemoteRLRuntimeError):
    pass

# ---------------------------------------------------------------------------
# 6) QUO  – Resource limit & quota
# ---------------------------------------------------------------------------
class RemoteRLQuotaError(RemoteRLError):
    pass

class SimulatorLimitExceededError(RemoteRLQuotaError):
    pass

class WorkerLimitExceededError(RemoteRLQuotaError):
    pass

class EnvRunnerLimitExceededError(RemoteRLQuotaError):
    pass

# ---------------------------------------------------------------------------
# 7) MISC / fallback
# ---------------------------------------------------------------------------
class RemoteRLUnknownError(RemoteRLError):
    """Catch-all for *truly* unexpected SDK errors."""

    pass

# ------------------------------------------------------------
# Mapping from concrete SDK error types → (code, status, headline)
# headline has only general information, no details. The details will attach in runtime with the exception context. 
# ------------------------------------------------------------
_DASHBOARD_URL = "https://remoterl.com/user/dashboard"
_REPOSITORY_URL = "https://github.com/ccnets-team/remoterl/"

_REMOTERL_EXCEPTION_MAP = {
    # ── 1  Authentication / Account ──────────────────────────
    NoApiKeyError:              ("AUTH-001", 501, f"No RemoteRL API key found - provide a valid key.\n         {_DASHBOARD_URL}"),
    InvalidApiKeyError:         ("AUTH-002", 502, f"Invalid API key - generate a new key in the dashboard.\n         {_DASHBOARD_URL}"),
    ApiKeyExpiredError:         ("AUTH-003", 503, f"API key expired or disabled - generate a fresh key.\n         {_DASHBOARD_URL}"),
    DataCreditExhaustedError:   ("AUTH-004", 504, f"Data credit exhausted - upgrade plan or wait for next refill.\n         {_DASHBOARD_URL}"),
    SessionDeniedError:         ("AUTH-005", 505, f"Session denied - verify your account status or contact support.\n         {_DASHBOARD_URL}"),
    RemoteRLAuthError:          ("AUTH-000", 550, f"Authentication failed - verify key and credentials.\n         {_DASHBOARD_URL}"),
    # ── 2  Configuration / CLI flags ─────────────────────────
    InvalidFrameworkError:      ("CFG-001", 551, f"Invalid RL framework - use Gymnasium, Ray RLlib, or Stable-Baselines3. see more details at\n        {_DASHBOARD_URL}"),
    EnvNotFoundError:           ("CFG-002", 552, "Environment not found - check the environment ID."),
    InvalidNumWorkersError:     ("CFG-003", 553, "num_workers is invalid"),
    InvalidNumEnvRunnersError:  ("CFG-004", 554, "num_env_runners is invalid"),
    UnknownOptionError:         ("CFG-005", 555, "Unknown option - verify CLI flags or config keys."),
    DuplicateOptionError:       ("CFG-006", 556, "Duplicate option provided - using the last value."),
    RemoteRLConfigError:        ("CFG-000", 600, "Configuration error - review settings."),
    # ── 3  Dependencies / Environment ────────────────────────
    UnsupportedPythonVersionError: ("DEP-001", 601, f"Unsupported Python version - install a supported release. 3.9~3.12 is supported. See more details at\n        {_REPOSITORY_URL}"),
    GymnasiumMissingError:      ("DEP-002", 602, "Gymnasium not installed - install 'gymnasium'."),
    SB3MissingError:            ("DEP-003", 603, "Stable-Baselines3 not installed - install 'stable-baselines3'."),
    RLLibMissingError:          ("DEP-004", 604, "Ray RLlib not installed - install 'ray[rllib]'."),
    CleanRLMissingError:        ("DEP-005", 605, "CleanRL not installed - install 'cleanrl'."),
    PettingZooMissingError:     ("DEP-006", 606, "PettingZoo not installed - install 'pettingzoo'."),
    RemoteRLVersionError:       ("DEP-011", 611, "RemoteRL version mismatch - install a compatible version."),
    RemoteRLVersionMissingErorr:("DEP-012", 612, "RemoteRL version information missing - reinstall the package."),
    GymnasiumVersionError:      ("DEP-021", 621, "Gymnasium version mismatch with the installed RemoteRL version - install a compatible version."), 
    SB3VersionError:            ("DEP-022", 622, "Stable-Baselines3 version mismatch with the installed RemoteRL version - install a compatible version."),   
    RLLibVersionError:          ("DEP-023", 623, "Ray RLlib version mismatch with the installed RemoteRL version - install a compatible version."),
    RLLibraryMissingError:      ("DEP-026", 626, "RL library not installed. See details"),
    GymnasiumPatchError:        ("DEP-031", 631, "Gymnasium patch failed - reinstall Gymnasium and RemoteRL in a clean python/conda environment or contact support."),
    StableBaselines3PatchError: ("DEP-032", 632, "Stable-Baselines3 patch failed - reinstall Stable-Baselines3 and RemoteRL in a clean python/conda environment or contact support."),
    RayRLlibPatchError:         ("DEP-033", 633, "Ray RLlib patch failed - reinstall Ray[rllib] and RemoteRL in a clean python/conda environment or contact support."),
    RemoteRLDepencyError:       ("DEP-000", 650, f"Dependency error - check required Python packages and versions.\n         {_REPOSITORY_URL} or {_DASHBOARD_URL} for more information."), 
    # ── 4  Network / Relay connectivity ─────────────────────
    ServerUnreachableError:     ("NET-001", 651, "Cannot reach RemoteRL servers - check your network."),
    GlobalRedirectionError:     ("NET-002", 652, "Failed to redirect to the right regional server."),
    SimulatorSessionOpenError:  ("NET-003", 653, "This Simulator failed to open the session with connected trainer"), 
    TrainerSessionOpenError:    ("NET-004", 654, "This Trainer failed to open the session wth connected simulator"), 
    ServerReachFailedError:     ("NET-006", 656, "Failed to reach the server - verify connectivity."),
    ServerListenFailedError:    ("NET-007", 657, "Failed to listen on the server - verify connectivity."),
    SessionConnectionLostError: ("NET-008", 658, "Session connection lost - check network or server status."),
    RelayConnectionLostError:   ("NET-009", 659, "Relay connection lost - check network or server status."),
    RelayReconnectionFailError: ("NET-010", 660, "Reconnection failed - check network or server status."),
    RemoteRelayTimeoutError:    ("NET-012", 662, "Remote environment step timeout - increase timeout or check latency."),
    WorkerRelayTimeoutError:    ("NET-013", 663, "Worker step timeout - the env_runner on the connected simulator side may be unresponsive."),
    EnvRunnerRelayTimeoutError: ("NET-014", 664, "EnvRunner step timeout - the worker on the connected trainer side may be unresponsive.."),
    RemoteRLNetworkError:       ("NET-000", 700, "Network error - check your internet connection or RemoteRL server status."),
    # ── 5  Runtime / Training loop ───────────────────────────
    SimulatorStartupError:      ("RUN-001", 701, "Simulator instance startup failed."),
    SimulatorCrashedError:      ("RUN-002", 702, "Simulator instance crashed. see detail"),
    SimulatorSessionCrashedError:("RUN-003", 703, "Simulator instance crashed running a session."),
    TrainerStartupError:        ("RUN-006", 706, "Training instance startup failed."),
    TrainerCrashedError:        ("RUN-007", 707, "Training instance crashed. see details"), 
    TrainerSessionCrashedError: ("RUN-008", 708, "Training instance crashed running a session."),
    WorkerStartupError:         ("RUN-011", 711, "Spawned worker startup failed."),
    GymWorkerStartupError:      ("RUN-012", 712, "Spawned Gym worker startup failed."),
    RayWorkerStartupError:      ("RUN-013", 713, "Spawned Ray worker startup failed."),
    PettingZooWorkerStartupError: ("RUN-014", 714, "Spawned PettingZoo worker startup failed."),
    EnvRunnerStartupError:      ("RUN-016", 716, "Spawned env_runner startup failed."),
    WorkerCrashedError:         ("RUN-021", 721, "Spawned worker instance crashed."),
    GymWorkerCrashedError:      ("RUN-022", 722, "Spawned Gym worker instance crashed."),
    RayWorkerCrashedError:      ("RUN-023", 723, "Spawned Ray worker instance crashed."),
    PettingZooWorkerCrashedError: ("RUN-024", 724, "Spawned PettingZoo worker instance crashed."),
    EnvRunnerCrashedError:      ("RUN-026", 726, "Spawned env_runner instance crashed."),
    EnvRunnerEnvCreationError:  ("RUN-030", 730, "EnvRunner environment creation failed - ensure simualtor installed the environment that your trainer requested."),
    EnvMakeError:               ("RUN-031", 731, "Failed to make an environment."),
    EnvStepError:               ("RUN-032", 732, "Environment step crashed."),
    EnvResetError:              ("RUN-033", 733, "Environment reset crashed."),
    EnvCloseError:              ("RUN-034", 734, "Environment close crashed."),
    EnvObservationSpaceError:   ("RUN-035", 735, "Environment observation space error."),
    EnvActionSpaceError:        ("RUN-036", 736, "Environment action space error."),
    ShutdownSetupError:         ("RUN-041", 741, "Shutdown setup failed in the RemoteRL init()."),
    ShutdownCrashedError:       ("RUN-042", 742, "Shutdown process crashed."),
    RemoteRLRuntimeError:       ("RUN-000", 750, "RemoteRL runtime error - see details."),
    # ── 6  Resource limit & quota ────────────────────────────
    SimulatorLimitExceededError:("QUO-001", 751, f"Simulator limit reached - reduce simulators or request quota increase.\n         {_DASHBOARD_URL}"),
    WorkerLimitExceededError:   ("QUO-002", 752, f"Worker limit reached - reduce num_workers or ask quota increase.\n         {_DASHBOARD_URL}"), 
    EnvRunnerLimitExceededError:("QUO-003", 753, f"EnvRunner limit reached - decrease num_env_runners or ask quota increase.\n         {_DASHBOARD_URL}"),

    # General RemoteRLQuotaError for any uncaught quota exceptions
    RemoteRLQuotaError:         ("QUO-000", 800, f"RemoteRL quota error - check your resource limits and request an increase if needed.\n         {_DASHBOARD_URL}"),
    # ── 7  Miscellaneous / Signals ───────────────────────────
    
    # General RemoteRLUnknownError for any uncaught exceptions 
    RemoteRLUnknownError:       ("MISC-000", 850, f"Unknown error occurred - check etails.\n         {_REPOSITORY_URL}"),
}

try:
    import gymnasium as gym
    _REMOTERL_EXCEPTION_MAP.update({
        gym.error.Error: ("GYM-000", 900, "Gymnasium error"),
    })                           
except Exception:
    pass

try:
    import stable_baselines3
except Exception:
    pass

try:
    import ray
    import ray.exceptions
    _REMOTERL_EXCEPTION_MAP.update({
        ray.exceptions.RayError: ("RAY-000", 920, "Ray error"),
    })
except Exception:
    pass


_REMOTERL_EXCEPTION_MAP.update({
    # General RemoteRLError for any uncaught exceptions
    RemoteRLError: ("SDK-000", 500, "RemoteRLError - an uncaught RemoteRL SDK exception occurred."),
})                           

ALL_REMOTERL_EXCEPTIONS = tuple(_REMOTERL_EXCEPTION_MAP.keys())
