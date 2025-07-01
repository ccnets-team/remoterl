#!/usr/bin/env python3
"""gym.py – Step‑based Gymnasium backend for RemoteRL
================================================================
This revision removes the *episode* outer‑loop and instead runs for a fixed
number of **steps** (across all vectorised environments).  Many modern
Gymnasium environments auto‑reset when they reach a terminal state, so tracking
per‑episode returns can be misleading or brittle.  Counting raw environment
steps is deterministic and version‑agnostic.

Usage
-----
Pass ``total_steps`` instead of ``num_episodes`` when calling
``train_gym({...})``.  Defaults to **50 000** steps if omitted.

The live report now prints:
* **inst_fps** – instantaneous steps/sec (over the last interval)
* **avg_fps**  – average steps/sec since the start
* **steps**    – total environment steps executed so far
"""
from __future__ import annotations

import inspect
import time
from typing import Any, Dict

import numpy as np
import typer
import gymnasium as gym

###############################################################################
# Utility helpers
###############################################################################

def _filter_config(func, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return only the kwargs that *func* actually accepts."""
    allowed = {p for p in inspect.signature(func).parameters if p != "self"}
    return {k: v for k, v in cfg.items() if k in allowed}


def _create_env(env_id: str, num_envs: int, **env_kwargs):
    """Construct a SyncVectorEnv when *num_envs* > 1, else a single env."""
    if num_envs is not None and num_envs > 0:
        num_envs = int(num_envs)
        env = gym.make_vec(env_id, num_envs=num_envs, **_filter_config(gym.make_vec, env_kwargs))
        typer.echo("Vectorised env mode")
    else:
        env = gym.make(env_id, **_filter_config(gym.make, env_kwargs))
        typer.echo("Single env mode")
    return env

###############################################################################
# Roll‑out loop (step based)
###############################################################################

def _run_steps(env: gym.Env, *, num_envs: int, total_steps: int, fps_interval: float = 1.0):
    """Execute *total_steps* random actions and stream live FPS statistics."""
    obs, _ = env.reset()
    num_envs = int(num_envs or 1)
    returns = np.zeros(num_envs, dtype=float)
    global_steps = 0

    start_time = last_report = time.time()
    steps_at_last_report = 0

    while global_steps < total_steps:
        # ------------------------------------------------------------------
        # ①  Sample actions – Gymnasium vector envs support batched sampling.
        # ------------------------------------------------------------------
        actions = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(actions)

        # ------------------------------------------------------------------
        # ②  Handle rewards & auto‑reset environments transparently.
        # ------------------------------------------------------------------
        reward = np.asarray(reward, dtype=float).reshape(-1)
        returns += reward

        global_steps += num_envs

        # ------------------------------------------------------------------
        # ③  Periodic live stats (FPS) -------------------------------------
        # ------------------------------------------------------------------
        now = time.time()
        if now - last_report >= fps_interval:
            steps_since = global_steps - steps_at_last_report
            inst_fps = steps_since / (now - last_report + 1e-8)
            avg_fps = global_steps / (now - start_time + 1e-8)

            typer.echo(
                f"[STATS] inst_fps {inst_fps:6.1f} | avg_fps {avg_fps:6.1f} | "
                f"steps {global_steps:9d}/{total_steps}"
            )
            last_report = now
            steps_at_last_report = global_steps

    elapsed = time.time() - start_time
    return returns, elapsed

###############################################################################
# Public entry‑point
###############################################################################

def train_gym(hyperparams: Dict[str, Any]) -> Dict[str, Any]:
    """Run a random policy for *total_steps* and return summary stats.

    Recognised *hyperparams*
    ------------------------
    env_id         (str)   – Gymnasium environment id (default ``CartPole-v1``)
    num_envs       (int)   – Parallel environments (default **32**)
    total_steps    (int)   – Steps across **all** envs (default **50_000**)
    fps_interval   (float) – Seconds between FPS reports (default **1.0**)

    All other keys are forwarded to ``gym.make`` / ``gym.make_vec``.
    """
    # ---------------------------------------------------------------------
    # 1️⃣  Parameter parsing & defaults
    # ---------------------------------------------------------------------
    env_id = str(hyperparams.pop("env_id", hyperparams.get("env", "CartPole-v1")))
    num_envs = hyperparams.pop("num_envs", 32)
    total_steps = int(hyperparams.pop("total_steps", 50_000))
    fps_interval = float(hyperparams.pop("fps_interval", 1.0))

    # ---------------------------------------------------------------------
    # 2️⃣  Environment construction
    # ---------------------------------------------------------------------
    env = _create_env(env_id, num_envs, **hyperparams)
    typer.echo(f"Env ID: {env_id}")
    typer.echo(f"Vector envs: {num_envs}")
    typer.echo(f"Total steps: {total_steps}")
    typer.echo(f"Observation space: {env.observation_space.shape}")
    try:
        typer.echo(f"Action space: {env.action_space.shape}")
    except AttributeError:
        typer.echo(f"Action space: {env.action_space}")

    # ---------------------------------------------------------------------
    # 3️⃣  Roll‑out
    # ---------------------------------------------------------------------
    returns, elapsed = _run_steps(env, num_envs=num_envs, total_steps=total_steps, fps_interval=fps_interval)

    # ---------------------------------------------------------------------
    # 4️⃣  Cleanup & summary
    # ---------------------------------------------------------------------
    env.close()

    overall_fps = total_steps / max(elapsed, 1e-9)
    summary = {
        "mean_reward": float(np.mean(returns) / (total_steps / num_envs)),  # per‑step mean
        "total_steps": total_steps,
        "overall_fps": overall_fps,
    }

    typer.echo(f"Run complete. Summary: {summary}")
    return summary

###############################################################################
# Quick smoke‑test -----------------------------------------------------------
###############################################################################
if __name__ == "__main__":  # pragma: no cover
    train_gym({})
