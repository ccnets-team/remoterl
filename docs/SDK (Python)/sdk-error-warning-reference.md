
| Code         | Headline (Console message) | Solution |
| ------------ | ------------------------- | -------- |
| **(AUTH-000)** | Authentication failed - verify key and credentials. |  |
| **(AUTH-001)** | No RemoteRL API key found. Pass valid key in RemoteRL init() function. | The SDK did not receive any API key, so it cannot authenticate with RemoteRL’s backend. — Supply your key in `remoterl.init(api_key="<YOUR_KEY>")` or set the `REMOTERL_API_KEY` environment variable. |
| **(AUTH-002)** | Invalid API key – authentication failed. Check your key or generate a new one in the dashboard. | The provided key is malformed or no longer recognised by the server. — Copy‑paste the key again (watch for hidden spaces); if the error persists, revoke the old key and create a fresh one on your dashboard. |
| **(AUTH-003)** | API key expired or disabled – generate a new key in the dashboard. |  |
| **(AUTH-004)** | Data credit exhausted – traffic gets slower and can pause. Upgrade your plan or wait for the next free data credit refill. | Your monthly data‑credit quota has reached zero; the service is throttling. — Wait for the first of next month or upgrade to a higher tier that includes more data credits. |
| Code         | Headline (Console message) | Solution |
| **(CFG-000)** | Configuration error - review settings. |  |
| **(CFG-001)** | Invalid RL framework – valid choices: Gymansium, Ray RLlib, Stable-Baselines3, … |  |
| **(CFG-002)** | Environment not found – check spelling or install the environment package. |  |
| **(CFG-003)** | num\_workers is invalid. | `num_workers` was < 1 or greater than your account limit. — Pass a positive integer ≤ your *Worker quota*; consult the dashboard for limits. |
| **(CFG-004)** | num\_env\_runners is invalid. | Same as above but for Env‑Runner processes. — Pick a value between 1 and the *EnvRunner quota*, or omit to let RemoteRL choose. |
| **(CFG-005)** | Unknown option – use --help for available flags. | The CLI (or init) received an unrecognised keyword argument. — Run `remoterl --help` or check the docs for valid flags; remove the typo. |
| **(CFG-006)** | Duplicate option provided – using the last value. | A config flag was specified twice; the last one wins. — Remove duplicates to avoid confusion. |
| Code         | Headline (Console message) | Solution |
| **(DEP-000)** | Dependency error - check required Python packages and versions. or dashboard for more information. |  |
| **(DEP-001)** | Unsupported Python version – RemoteRL requires Python 3.9–3.12. | You’re running an interpreter outside the supported range. — Upgrade (or create a venv) with a supported Python version. |
| **(DEP-002)** | Gymnasium not installed – install with `pip install gymnasium`. | The SDK imports Gymnasium by default; it’s missing. — `pip install gymnasium` (or the extra `gymnasium[all]` if you need Mujoco, Atari…). |
| **(DEP-003)** | Stable‑Baselines3 selected but the package is not installed. | You requested SB3 features but the library is absent. — `pip install stable‑baselines3[extra]` then retry. |
| **(DEP-004)** | RLlib selected but Ray RLlib is not installed. | Same for RLlib. — `pip install "ray[rllib]"` or match the CUDA / Python pin suggested in the docs. |
| **(DEP-005)** | Gymnasium version mismatch – please check available versions… | The installed Gymnasium is newer/older than the version bundle RemoteRL patched against. — Pin to the recommended version shown in the docs, e.g. `pip install gymnasium==0.29.0`. |
| **(DEP-006)** | Stable‑Baselines3 version mismatch… | Same as above for SB3. — Downgrade or upgrade SB3 to the supported release. |
| **(DEP-011)** | RemoteRL version mismatch - install a compatible version. |  |
| **(DEP-012)** | RemoteRL version information missing - reinstall the package. | |
| **(DEP-021)** | Gymnasium version mismatch with the installed RemoteRL version - install a compatible version. |  |
| **(DEP-022)** | Stable-Baselines3 version mismatch with the installed RemoteRL version - install a compatible version. |  |
| **(DEP-023)** | Ray RLlib version mismatch with the installed RemoteRL version - install a compatible version. |  |
| **(DEP-026)** | RL library not installed. See details |  |
| **(DEP-031)** | Gymnasium patch failed - reinstall Gymnasium and RemoteRL in a clean python/conda environment or contact support. |  |
| **(DEP-032)** | Stable-Baselines3 patch failed - reinstall Stable-Baselines3 and RemoteRL in a clean python/conda environment or contact support. |  |
| **(DEP-033)** | Ray RLlib patch failed - reinstall Ray[rllib] and RemoteRL in a clean python/conda environment or contact support. |  |
| Code         | Headline (Console message) | Solution |
| **(NET-000)** | Network error - check your internet connection or RemoteRL server status. |  |
| **(NET-001)** | Cannot reach RemoteRL servers – check network. | Initial Web‑Socket handshake failed (DNS, firewall, proxy, etc.) |
| **(NET-002)** | Failed to redirect to the right RemoteRL server. |  |
| **(NET-003)** | This simulator failed in matching with available trainers. | |
| **(NET-004)** | This trainer failed in matching with available simulators. |  |
| **(NET-006)** | Failed to reach the server – check network or server status. |  |
| **(NET-007)** | Failed to listen on the server – check network or server status. |  |
| **(NET-008)** | Lost connection to session – check network or session status. |  |
| **(NET-009)** | Lost connection in data relay – check network. |  |
| **(NET-010)** | Failed to reconnect to the server – check network or server status. |  |
| **(NET-012)** | Remote env step took over the timeout limit. |  |
| **(NET-013)** | Remote worker step took over the timeout limit – the env\_runner on the simulator side may be unresponsive. |  |
| **(NET-014)** | Remote env\_runner step took over the timeout limit – the worker on the trainer side may be unresponsive. |  |
| Code         | Headline (Console message) | Solution |
| **(RUN-000)** | RemoteRL runtime error - see details. |  |
| **(RUN-001)** | Training instance crashed. |  |
| **(RUN-002)** | Training session error. |  |
| **(RUN-003)** | Simulator instance crashed. | |
| **(RUN-006)** | Spawned Ray worker init failed. |  |
| **(RUN-007)** | Spawned env\_runner init failed. |  |
| **(RUN-008)** | Spawned worker instance crashed. |  |
| **(RUN-011)** | Spawned env\_runner instance crashed. |  |
| **(RUN-012)** | Spawned Gym worker startup failed. |  |
| **(RUN-013)** | Spawned Ray worker startup failed. |  |
| **(RUN-014)** | Spawned PettingZoo worker startup failed. |  |
| **(RUN-016)** | Spawned env_runner startup failed. |  |
| **(RUN-021)** | Spawned worker instance crashed. |  |
| **(RUN-022)** | Spawned Gym worker instance crashed. | |
| **(RUN-023)** | Environment step crashed. |  |
| **(RUN-024)** | Environment reset crashed. |  |  |
| **(RUN-030)** | Shutdown process crashed. |  |
| **(RUN-031)** | Failed to make an environment. |  |
| **(RUN-032)** | Environment step crashed. |  |
| **(RUN-033)** | Environment reset crashed. |  |
| **(RUN-034)** | Environment close crashed. |  |
| **(RUN-035)** | Environment observation space error. |  |
| **(RUN-036)** | Environment action space error. |  |
| **(RUN-041)** | Shutdown setup failed in the RemoteRL init(). |  |
| **(RUN-042)** | Shutdown process crashed. |  |
| Code         | Headline (Console message) | Solution |
| **(QUO-000)** | RemoteRL quota error - check your resource limits and request an increase if needed.\n dashboard |  |
| **(QUO-001)** | Simulator limit reached – decrease simulators or request a quota increase. |  |
| **(QUO-002)** | Worker limit reached – decrease num\_workers or request a quota increase. |  |
| **(QUO-003)** | EnvRunner limit reached – decrease num\_env\_runners or request a quota increase. |  |
| Code         | Headline (Console message) | Solution |
| **(MISC-000)** | Unknown error occurred - check etails. |  |
| Code         | Headline (Console message) | Solution |
| **(SDK-000)** | Unknown General Error in RemoteRL - check etails. |  |