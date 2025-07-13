# RemoteRL SDK Error & Warning Reference — Systematic Error Handling

## What is this document? 

We catalog every exception the RemoteRL Python SDK can raise—whether you call it from your own Python code or run it via our command-line interface (CLI)—in this document. Consolidating them in one place improves diagnostics and user experience by ensuring each exception carries rich runtime context.

### Example

If an authentication error occurs, the SDK raises `InvalidApiKeyError` with a message like:

```
[RemoteRL] (AUTH-002) Invalid API key – generate a new one
(details: server reported no key found for this account)
```

## Error Handling in the SDK

There are two types of error messages in the SDK; Fatal errors and warnings.

For fatal `errors`, the SDK raises an exception that stops the execution of the program and prints the error message to the console. The error message includes a error code, a headline, and contextual details that help diagnose the issue.

For `warnings`, the SDK logs a message to the console but does not stop the execution.


### Error message anatomy

Every error message follows a consistent format:
```
[RemoteRL] (error code) headline…
(details: …)
```

### Seven high‑level families

All exceptions belongs to one of the seven high‑level families that helps to understand about the error quickly.

* **AUTH** – authentication & authorisation problems
* **CFG**  – invalid configuration or missing settings
* **DEP**  – optional‑dependency issues (Gymnasium, Ray, SB3, …)
* **NET**  – network and transport failures
* **RUN**  – runtime errors inside an environment or worker
* **QUO**  – quota or rate‑limit violations
* **MISC** – anything that does not fit the above (fallback)


### Error Codes

Each family has sub‑categories that provide more specific information about the error. For example, the `AUTH` family has an error code like `AUTH-001` to `AUTH-005`, telling more specific issues related to authentication, such as "No RemoteRL API key found" or "API key expired or disabled".


### Contextual details

The *details* are attached at runtime with the exception context to customise the message and provide more information about the error.


## Roadmap

We plan to capture exception details more effectively and improve exception identification by providing better context:

```
RemoteRLError (catch‑all)
    → 7 high‑level families (catch all by family)
        → 50+ specific error codes (catch-all by code, with contextual details)
```

The headline is the main message that appears in the console; it may provide enough information, but for more detail—especially on unhandled issues in the released SDK—we will soon add a `How to fix` section with actionable steps for each error.


|     Code     |     Headline (Console message)     |     How to fix     |
|------|---------|------------|
|     AUTH     |   Authentication Issue     |         |
| `AUTH-000` | Authentication failed - verify key and credentials. |  |
| `AUTH-001` | No RemoteRL API key found - provide a valid key. |  |
| `AUTH-002` | Invalid API key - generate a new key in the dashboard. |  |
| `AUTH-003` | API key expired or disabled - generate a fresh key. |  |
| `AUTH-004` | Data credit exhausted - upgrade plan or wait for next refill. |  |
| `AUTH-005` | Session denied - verify your account status or contact support. |  |
|     CFG     |   Configuration Issue      |         |
| `CFG-000` | Configuration error - review settings. |  |
| `CFG-001` | Invalid RL framework - use Gymnasium, Ray RLlib, or Stable-Baselines3. |  |
| `CFG-002` | Environment not found - check the environment ID. |  |
| `CFG-003` | num_workers is invalid |  |
| `CFG-004` | num_env_runners is invalid |  |
| `CFG-005` | Unknown option - verify CLI flags or config keys. |  |
| `CFG-006` | Duplicate option provided - using the last value. |  |
|     DEP     |     Dependency Issue     |          |
| `DEP-000` | Dependency error - check required Python packages and versions. or for more information. |  |
| `DEP-001` | Unsupported Python version - install a supported release. 3.9~3.12 is supported. |  |
| `DEP-002` | Gymnasium not installed - install 'gymnasium'. |  |
| `DEP-003` | Stable-Baselines3 not installed - install 'stable-baselines3'. |  |
| `DEP-004` | Ray RLlib not installed - install 'ray[rllib]'. |  |
| `DEP-005` | CleanRL not installed - install 'cleanrl'. |  |
| `DEP-006` | PettingZoo not installed - install 'pettingzoo'. |  |
| `DEP-011` | RemoteRL version mismatch - install a compatible version. |  |
| `DEP-012` | RemoteRL version information missing - reinstall the package. |  |
| `DEP-021` | Gymnasium version mismatch with the installed RemoteRL version - install a compatible version. |  |
| `DEP-022` | Stable-Baselines3 version mismatch with the installed RemoteRL version - install a compatible version. |  |
| `DEP-023` | Ray RLlib version mismatch with the installed RemoteRL version - install a compatible version. |  |
| `DEP-026` | RL library not installed. |  |
| `DEP-031` | Gymnasium patch failed - reinstall Gymnasium and RemoteRL in a clean python/conda environment or contact support. |  |
| `DEP-032` | Stable-Baselines3 patch failed - reinstall Stable-Baselines3 and RemoteRL in a clean python/conda environment or contact support. |  |
| `DEP-033` | Ray RLlib patch failed - reinstall Ray[rllib] and RemoteRL in a clean python/conda environment or contact support. |  |
|     NET     |     Network Issue     |          |
| `NET-000` | Network error - check your internet connection or RemoteRL server status. |  |
| `NET-001` | Cannot reach RemoteRL servers - check your network. |  |
| `NET-002` | Failed to redirect to the right regional server. |  |
| `NET-003` | This Simulator failed to open the session with connected trainer |  |
| `NET-004` | This Trainer failed to open the session wth connected simulator |  |
| `NET-006` | Failed to reach the server - verify connectivity. |  |
| `NET-007` | Failed to listen on the server - verify connectivity. |  |
| `NET-008` | Session connection lost - check network or server status. |  |
| `NET-009` | Relay connection lost - check network or server status. |  |
| `NET-010` | Reconnection failed - check network or server status. |  |
| `NET-012` | Remote environment step timeout - increase timeout or check latency. |  |
| `NET-013` | Worker step timeout - the env_runner on the connected simulator side may be unresponsive. |  |
| `NET-014` | EnvRunner step timeout - the worker on the connected trainer side may be unresponsive.. |  |
|     RUN     |     Runtime Issue     |          |
| `RUN-000` | RemoteRL runtime error |  |
| `RUN-001` | Simulator instance startup failed. |  |
| `RUN-002` | Simulator instance crashed. see detail |  |
| `RUN-003` | Simulator instance crashed running a session. |  |
| `RUN-006` | Training instance startup failed. |  |
| `RUN-007` | Training instance crashed. |  |
| `RUN-008` | Training instance crashed running a session. |  |
| `RUN-011` | Spawned worker startup failed. |  |
| `RUN-012` | Spawned Gym worker startup failed. |  |
| `RUN-013` | Spawned Ray worker startup failed. |  |
| `RUN-014` | Spawned PettingZoo worker startup failed. |  |
| `RUN-016` | Spawned env_runner startup failed. |  |
| `RUN-021` | Spawned worker instance crashed. |  |
| `RUN-022` | Spawned Gym worker instance crashed. |  |
| `RUN-023` | Spawned Ray worker instance crashed. |  |
| `RUN-024` | Spawned PettingZoo worker instance crashed. |  |
| `RUN-026` | Spawned env_runner instance crashed. |  |
| `RUN-030` | EnvRunner environment creation failed - ensure simualtor installed the environment that your trainer requested. |  |
| `RUN-031` | Failed to make an environment. |  |
| `RUN-032` | Environment step crashed. |  |
| `RUN-033` | Environment reset crashed. |  |
| `RUN-034` | Environment close crashed. |  |
| `RUN-035` | Environment observation space error. |  |
| `RUN-036` | Environment action space error. |  |
| `RUN-041` | Shutdown setup failed in the RemoteRL init(). |  |
| `RUN-042` | Shutdown process crashed. |  |
|     QUO     |     Quota Issue     |          |
| `QUO-000` | RemoteRL quota error - check your resource limits and request an increase if needed. |  |
| `QUO-001` | Simulator limit reached - reduce simulators or request quota increase. |  |
| `QUO-002` | Worker limit reached - reduce num_workers or ask quota increase. |  |
| `QUO-003` | EnvRunner limit reached - decrease num_env_runners or ask quota increase. |  |
|     MISC     |     Miscellaneous     |          |
| `MISC-000` | Unknown error occurred - check etails. |  |
|     GYM     |     Gymnasium Issue     |          |
| `GYM-000` | Gymnasium error |  |
|     RAY     |     Ray Rllib Issue     |          |
| `RAY-000` | Ray error |  |
|     SDK     |     SDK Issue     |          |
| `SDK-000` | RemoteRLError - an uncaught RemoteRL SDK exception occurred. |  |