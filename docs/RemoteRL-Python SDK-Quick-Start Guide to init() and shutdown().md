# Using `remoterl.init()` and `remoterl.shutdown()` in RemoteRL

RemoteRL is a lightweight Python SDK that lets you run reinforcement-learning (RL) **trainers** and **simulators** on separate machines without changing your core RL code.
You only need two public calls:

* **`remoterl.init()`** — start a remote session (trainer **or** simulator)
* **`remoterl.shutdown()`** — cleanly tear it down

The rest of your RL workflow (Gymnasium, Stable-Baselines3, RLlib, etc.) stays the same. It works on python scripts, Jupyter notebooks, AWS SageMaker instances, or any Python environment.

---

## `remoterl.init()` — Initialize a Remote Session

### Purpose

Call **once at program start**.

* **Trainer (`role="trainer"`)**

  * Spawns local worker processes.
  * Patches Gymnasium / Stable-Baselines3 / RLlib so every `gym.make()` or vector-env request talks to remote simulators.

* **Simulator (`role="simulator"`)**

  * Registers this process as an environment host **and blocks**, waiting for trainers to connect and request environments.

### Function signature

```python
remoterl.init(
    api_key: Optional[str],
    role: Literal["trainer", "simulator"],
    *,
    num_workers: int = 1,
    num_env_runners: int = 2,
) -> bool
```

### Parameters

| Name              | Type / Default               | Applies to   | Description                                                                                                                        |
| ----------------- | ---------------------------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| `api_key`         | `str \| None`                | both         | RemoteRL Cloud key (pass `None` for on-prem hubs). Can also be supplied via the `REMOTERL_API_KEY` env var.                        |
| `role`            | `"trainer"` \| `"simulator"` | both         | Determines whether this process performs training or hosts environments.                                                           |
| `num_workers`     | `int`, default = `1`         | trainer only | **Total** environment-runner tasks *across* all simulators. If runners > simulators, some simulators get multiple runners.<br>*(Ignored when using RLlib—RLlib manages its own rollout workers. We use what RLlib provides in this case)*     |
| `num_env_runners` | `int`, default = `2`         | trainer only | **Number of environment-runner tasks per simulator.** If runners > simulators, each simulator may host multiple runners.           |


### Return value

`True` if the session connects successfully; `False` if the backend or simulators are unreachable.
A `RuntimeError` is raised for fatal issues (e.g. invalid API key).

### What happens internally?

#### Trainer flow

1. **Connect** to the RemoteRL backend with the supplied `api_key`.
2. **Spawn** `num_workers` processes and allocate up to `num_env_runners` async tasks.
3. **Patch** Gymnasium / RLlib / Stable-Baselines3 so subsequent environment calls transparently use remote simulators.

#### Simulator flow

1. **Connect** to the backend and register as a simulator.
2. **Block** inside `init()`; wait indefinitely until trainers request work.
3. **Create & step environments** locally as commanded, returning observations/rewards over the wire.

### Minimal examples

#### Minimal trainer

```python
import remoterl
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

remoterl.init(
    api_key="YOUR_KEY",
    role="trainer",
    num_workers=1,
    num_env_runners=8,   # adjust for your workload
)

env   = make_vec_env("CartPole-v1", n_envs=32)   # actually runs on remote simulators
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

remoterl.shutdown()
```

#### Minimal simulator

```python
import remoterl

remoterl.init(api_key="YOUR_KEY", role="simulator")   # blocks here
# Press Ctrl-C or send a signal to stop:
remoterl.shutdown()
```

> **Tip** Calling `remoterl.shutdown()` restores Gymnasium / RLlib / SB3 to local mode, so you can drop back to local execution inside the same Python process if needed.

## Summary: Typical Workflow

To put it all together, here’s a quick summary of how you would use `init()` and `shutdown()` in a RemoteRL workflow:

1. **Start one or more simulators** (each on a separate machine or process as needed) by running a simulator script that calls `remoterl.init(..., role="simulator")`. These will connect to the RemoteRL service and wait for work.
2. **Start your trainer** script or program. At the beginning, call `remoterl.init(..., role="trainer")` with the same API key. Check that it returns True (connected). If not, handle the error (e.g., no simulators available or network issue).
3. Proceed with **creating environments and training your RL agent**. With RemoteRL’s integration, your environment creation and training calls (Gymnasium, Stable-Baselines3, RLlib, etc.) will automatically utilize the remote simulators. The trainer will send actions to the simulators and receive observations/rewards, all transparently.
4. Once training is complete (or if you need to halt early), call `remoterl.shutdown()`(not mandetory) to **end the remote session**. This stops background workers and cleans up on the trainer side.  
   *If you are terminating the Python process anyway (for example, the script ends or you call `ray.shutdown()`), an explicit call is not required—RemoteRL cleans up automatically.*

---
