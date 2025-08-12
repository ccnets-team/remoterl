#!/usr/bin/env python3
"""
Step-based trainer for ``StockTradingVecEnv``.

- Connect to RemoteRL as a trainer.
- Create a vectorized environment with ``num_envs`` slots.
- Run a fixed number of environment steps (step-based rather than episode-based).
- Periodically print instantaneous and average FPS, followed by a summary.
"""

from __future__ import annotations

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import typer
import time
import numpy as np
import gymnasium as gym
import remoterl

# ---------------- Configuration ----------------
REMOTERL_API_KEY = os.getenv("REMOTERL_API_KEY")
ENV_ID = "stock-trading-vec-env"

NUM_ENVS = int(os.getenv("NUM_ENVS", "32"))        # number of parallel environment slots
TOTAL_STEPS = int(os.getenv("TOTAL_STEPS", "50000"))  # total environment steps across all slots
FPS_INTERVAL = float(os.getenv("FPS_INTERVAL", "1.0"))  # reporting interval in seconds

# ----------------------------------------------


def main() -> None:
    # 1) Connect as a trainer
    if not remoterl.init(api_key=REMOTERL_API_KEY, role="trainer", num_env_runners=1):
        return

    # 2) Build the vectorized environment
    env = gym.make_vec(ENV_ID, num_envs=NUM_ENVS)

    # 3) Step-based rollout (properly handles vector rewards and terminations)
    obs, _ = env.reset()

    returns = np.zeros(NUM_ENVS, dtype=np.float64)  # cumulative reward per environment
    global_steps = 0

    start_time = last_report = time.time()
    steps_at_last_report = 0

    while global_steps < TOTAL_STEPS:
        actions = env.action_space.sample()                     # shape: (NUM_ENVS,)
        obs, reward, terminated, truncated, info = env.step(actions)

        typer.echo(f"observation: {obs} at step {global_steps}")

        # Process vectorized outputs
        reward = np.asarray(reward, dtype=np.float64).reshape(-1)        # (NUM_ENVS,)
        terminated = np.asarray(terminated, dtype=bool).reshape(-1)      # (NUM_ENVS,)
        truncated  = np.asarray(truncated, dtype=bool).reshape(-1)       # (NUM_ENVS,)

        returns += reward
        global_steps += NUM_ENVS

        # Auto-resets are handled by Gymnasium's vector environments; no manual reset needed

        # Periodic FPS logging
        now = time.time()
        if now - last_report >= FPS_INTERVAL:
            steps_since = global_steps - steps_at_last_report
            inst_fps = steps_since / max(now - last_report, 1e-8)
            avg_fps  = global_steps / max(now - start_time, 1e-8)
            typer.echo(f"[STATS] inst_fps {inst_fps:6.1f} | avg_fps {avg_fps:6.1f} | "
                       f"steps {global_steps:9d}/{TOTAL_STEPS}")
            last_report = now
            steps_at_last_report = global_steps

    elapsed = time.time() - start_time
    overall_fps = TOTAL_STEPS / max(elapsed, 1e-9)

    # Optional performance summary
    mean_reward_per_step = float(np.mean(returns) / (TOTAL_STEPS / NUM_ENVS))
    summary = {
        "mean_reward_per_step": mean_reward_per_step,
        "total_steps": TOTAL_STEPS,
        "num_envs": NUM_ENVS,
        "overall_fps": overall_fps,
        "elapsed_sec": elapsed,
    }
    print(f"Run complete. Summary: {summary}")

    env.close()


if __name__ == "__main__":
    main()
