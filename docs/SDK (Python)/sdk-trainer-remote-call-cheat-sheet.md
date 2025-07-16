# SDK (Python) Trainer Remote Call Cheat‑sheet

**Audience:** Trainers who have already called `remoterl.init(api_key, role="trainer")`.

> Keep writing normal Gym, Stable‑Baselines3, or RLlib code; RemoteRL silently swaps in remote simulators.

---

#### 1 Library‑by‑Library reference

##### Universal Gymnasium Based Remote Calls

* **`gymnasium.make("env_id")`, `gymnasium.make_vec(...)`** → returns a `RemoteEnv` proxy; every `reset`, `step`, and `close` call is routed through the gateway, with parallelism governed by the `num_workers` parameter you passed to `remoterl.init()`.

### Gymnasium / Gym

| Call                                         | RemoteRL behaviour                                           | Notes                                                            |
| -------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------------- |
| `gymnasium.make("env_id")`                   |                                                              |                                                                  |
| `gym.make(...)`                              | Returns **`RemoteEnv` proxy** (same behaviour as `gym.make()`)                               | `reset`, `step`, `close` calls are forwarded through the gateway |
| `gym.make_vec(...)`                              | Returns **`RemoteEnvVec`  proxy** (same behaviour as `gym.make_vec()`)                                | `reset`, `step`, `close` calls are remote same as `gym.make()` |
| `gym.register()`                              | **Register** env on remote simulators **                                |  |
| `env.render()`                               | Placeholder – frame streaming will arrive with **SDK 1.2.0** |                                                                  |
| `env.observation_space` / `env.action_space` | Fetched once from the simulator then cached locally          |                                                                  |


### Stable‑Baselines3

| Call / Helper                           | RemoteRL behaviour                                                         | Notes                                              |
| --------------------------------------- | -------------------------------------------------------------------------- | -------------------------------------------------- |
| `sb3.common.env_util.make_vec_env(...)` | Patched to build **remote** vector envs by default                         | Uses the same Gym patched factories under the hood |
| `DummyVecEnv`         | Public API unchanged; batched `step()` and `reset()` fan‑out to simulators |                                                    |
| `SubprocVecEnv`         | Same as `DummyVecEnv` |    `n_envs` is used as remoterl `num_workers` and `num_env_runners` in the subprocesses                                                |

### Ray RLlib

| Call / Helper                      | RemoteRL behaviour                                                                        | Notes                                                            |
| ---------------------------------- | ----------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `tune.register_env`                | Mirrors every registration on the gateway                                                 |                             |
| `AlgorithmConfig.build_algo()`                   | Build your algo as usual; each env runners proxy to simulator via cloud server | 


---


#### 3 Mental model

```
Trainer  ──(WebSocket)──▶ Cloud Server  ──(WebSocket)──▶ Simulator(s)
    --->        --->         --->        --->         [action] 


Trainer  ◀──(WebSocket)─── Cloud Server ◀──(WebSocket)─── Simulator(s)
      [obs, reward, term, truc, info]   <-----   <-----   <-----         

```


*Last updated: 2025‑07‑07*
