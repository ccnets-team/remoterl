#!/usr/bin/env python3
"""sb3.py - Version-free Stable-Baselines3 (SB3) integration via typed-config CLI parsing.

This module enables training an RL agent with Stable-Baselines3 through a unified 
dictionary-based API (`train_sb3`). It provides default hyperparameters and 
helper functions to mirror the style of the RLlib integration, ensuring a 
consistent interface across different RL frameworks. By using internal helpers 
(like `filter_config` and `ensure_default_hyperparams`) similar to those in 
`rllib.py`, the SB3 training pipeline remains easy to use and maintain.

**Usage:** Prepare a dictionary of hyperparameters and call `train_sb3(hyperparams)`.
The function will set up the Gymnasium environment(s), configure the SB3 algorithm 
(e.g., PPO, DQN) with the given settings, train the model for the specified number 
of timesteps, and save the resulting model to disk.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import typer

try:
    import stable_baselines3 as sb3
except ModuleNotFoundError as err:  # pragma: no cover
    # If SB3 is not installed, provide a clear error with installation instructions.
    raise ModuleNotFoundError(
        "RL Framework 'sb3' selected but *stable-baselines3* is not installed.\n"
        "Install it with:\n\n"
        "    pip install stable-baselines3[extra]\n"
    ) from err

from .helpers import filter_config

# ----------------------------------------------------------------------------- 
# 1. Utility helpers 
# ----------------------------------------------------------------------------- 

def guess_ppo_hyperparams(n_envs: int, max_episode_steps: int, target_rollout: int = 2048) -> tuple[int, int]:
    """Heuristic to choose PPO's n_steps and n_epochs based on env length and parallelism.

    For PPO algorithms, this function computes a reasonable pair of `n_steps` 
    (number of timesteps per environment per update) and `n_epochs` (number of 
    epochs to train on each batch) given the number of parallel environments and 
    the typical episode length. The goal is to have an overall rollout of roughly 
    `target_rollout` timesteps per update, with adjustments for very short or long episodes.

    Args:
        n_envs (int): Number of parallel environments.
        max_episode_steps (int or None): The maximum steps per episode for the environment 
            (if None, a default of 1000 steps is assumed).
        target_rollout (int): Desired total number of timesteps in one rollout (default 2048).

    Returns:
        tuple[int, int]: A tuple `(n_steps, n_epochs)` recommended for PPO training.
    """
    # Ensure we have integer values for calculations.
    ep_len = max_episode_steps if max_episode_steps is not None else 1000
    ep_len = int(ep_len)
    n_envs = int(n_envs or 1)
    target_rollout = int(target_rollout)

    # Determine n_steps based on episode length:
    # - For very short episodes (<= 50 steps), allow up to 5 episodes per rollout (capped at 2048).
    # - For extremely long episodes (>= 3000 steps), use a smaller n_steps to avoid huge batches.
    # - Otherwise, choose n_steps so that n_steps * n_envs is around target_rollout (at least 16).
    if ep_len <= 50:
        n_steps = min(5 * ep_len, 2048)
    elif ep_len >= 3000:
        n_steps = 1024
    else:
        n_steps = max(16, target_rollout // n_envs)

    # Total rollout = n_steps * n_envs. Determine n_epochs based on how large the batch is:
    # - If total rollout is very large (>= 4096), fewer epochs (3) to limit training time per update.
    # - If moderately large (>= 2048), use 5 epochs.
    # - Otherwise (smaller batches), use more epochs (8) for more gradient updates per batch.
    rollout = n_steps * n_envs
    if rollout >= 4096:
        n_epochs = 3
    elif rollout >= 2048:
        n_epochs = 5
    else:
        n_epochs = 8

    return n_steps, n_epochs

def ensure_default_hyperparams(hp: Dict[str, Any]) -> Dict[str, Any]:
    """Merge user hyperparams with SB3 defaults and auto-tune PPO settings if needed.

    This function fills in default values for missing SB3 hyperparameters and 
    normalizes certain entries. If the selected algorithm is PPO and `n_steps` or 
    `n_epochs` are not provided, it automatically computes them using 
    `guess_ppo_hyperparams` to tailor the rollout size to the environment.

    Args:
        hp (Dict[str, Any]): Partial hyperparameters provided by the user.

    Returns:
        Dict[str, Any]: A complete hyperparameters dictionary with defaults applied.
    """
    # Base default hyperparameters for SB3 training.
    env_val = (
        hp.pop("env_id", None)                # current RLlib config key for env
        or hp.pop("env", None)    # accept 'env_id' for backward compatibility (remove it if present)
        or "CartPole-v1"                      # default environment if none provided
    )
        
    defaults = {
        "env_id": env_val,
        "n_envs": 32,
        "algo": "PPO",
        "policy": "MlpPolicy",
        "total_timesteps": 100_000,
        # ----- PPO-specific defaults -----
        "batch_size": 64,
        "gamma": 0.99,
        "learning_rate": 3e-4,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        # ----- Misc defaults -----
        "device": "auto",
        "verbose": 1,
        "log_dir": str(Path.cwd() / "sb3_logs"),
    }

    # Combine user-provided hyperparams with defaults. User values override defaults.
    merged = {**defaults, **(hp or {})}

    # Auto-tune n_steps and n_epochs for PPO if the user didn't specify them.
    if merged.get("algo", "PPO").upper() == "PPO":
        if merged.get("n_steps") is None or merged.get("n_epochs") is None:
            # Use a nominal episode length (1000) as guess if actual length unknown.
            ep_len_guess = 1000
            n_s, n_e = guess_ppo_hyperparams(merged.get("n_envs", 1), ep_len_guess)
            merged.setdefault("n_steps", n_s)
            merged.setdefault("n_epochs", n_e)

    # Normalize certain fields:
    merged["algo"] = merged["algo"].upper()                # algorithm name in uppercase (e.g., "PPO")
    merged["n_envs"] = int(merged["n_envs"])               # ensure n_envs is int
    merged["total_timesteps"] = int(merged["total_timesteps"])  # ensure timesteps is int

    # Drop any keys where value is None to avoid passing None into SB3 functions.
    merged = {k: v for k, v in merged.items() if v is not None}
    return merged

# ----------------------------------------------------------------------------- 
# 2. Algorithm registry 
# ----------------------------------------------------------------------------- 

# Mapping from algorithm name to SB3 BaseAlgorithm class for quick lookup.
_ALGOS: Dict[str, Any] = {
    "PPO": sb3.PPO,
    "A2C": sb3.A2C,
    "DQN": sb3.DQN,
    "SAC": sb3.SAC,
    "TD3": sb3.TD3,
    "DDPG": sb3.DDPG,
}

# ----------------------------------------------------------------------------- 
# 3. Main entry-point for SB3 training 
# ----------------------------------------------------------------------------- 

def train_sb3(hyperparams: Dict[str, Any]) -> None:
    """Train an RL agent using Stable-Baselines3 with the given hyperparameters.

    This is the primary entry point for SB3 training in the RemoteRL SDK. It 
    accepts a dictionary of hyperparameters, sets up the environment and algorithm, 
    runs the training loop, and saves the trained model to disk. The interface 
    is designed to be similar to RLlib's trainer workflow, but for SB3.

    Args:
        hyperparams (Dict[str, Any]): A dictionary of hyperparameters and settings. 
            This can include SB3-specific parameters (e.g., 'learning_rate', 'gamma') 
            as well as convenience keys like 'env_id', 'n_envs', 'algo', etc. 
            Unknown keys are filtered out and ignored.

    Returns:
        None: This function trains the model and saves it to a file (e.g., 
        "ppo_CartPole-v1.zip"). There is no direct return value.
    """
    # Apply default values and ensure required hyperparams are set (without mutating the original dict).
    hyperparams = ensure_default_hyperparams((hyperparams or {}).copy())

    # ------------------------------------------------------------------ 
    # 1) Environment construction 
    # ------------------------------------------------------------------ 
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv
    # Prepare arguments for environment creation, filtering out unrelated keys.
    # hyperparams["vec_env_cls"] = SubprocVecEnv  # Optional: use SubprocVecEnv for parallelism 
    vec_env_kwargs = filter_config(make_vec_env, hyperparams)
    # Create the Gymnasium environment. If n_envs > 1, this creates a vectorized env with n_envs copies.
    typer.echo(f"[INFO] Creating environment with parameters: {vec_env_kwargs}")
    env = make_vec_env(**vec_env_kwargs)

    # ------------------------------------------------------------------ 
    # 2) Algorithm selection and model initialization 
    # ------------------------------------------------------------------ 
    # Select the SB3 algorithm class based on 'algo' hyperparam (default to PPO if algo not recognized).
    algo_cls = _ALGOS.get(str(hyperparams.get("algo")).upper(), sb3.PPO)
    # Filter hyperparams for those accepted by the algorithm constructor (e.g., learning_rate, etc.).
    model_cfg = filter_config(algo_cls, hyperparams)
    # Use specified policy (e.g., "MlpPolicy"), or default to "MlpPolicy" if not provided.
    policy = model_cfg.pop("policy", "MlpPolicy")
    typer.echo(f"[INFO] Using algorithm: {algo_cls.__name__} with policy: {policy} and config: {model_cfg}")
    # Instantiate the SB3 model with the selected algorithm, policy, environment, and remaining config.
    model = algo_cls(policy=policy, env=env, **model_cfg)

    # ------------------------------------------------------------------ 
    # 3) Logging setup 
    # ------------------------------------------------------------------ 
    from stable_baselines3.common.logger import configure as sb3_configure
    if "log_dir" in hyperparams:
        log_dir = Path(hyperparams["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Configure SB3 logger to log to stdout and CSV files in the specified directory.
            model.set_logger(sb3_configure(str(log_dir), ["stdout", "csv"]))
        except Exception as exc:
            # If SB3's logging configuration fails (e.g., older SB3 version), log the error and continue.
            typer.echo(f"[ERROR] Failed to set up logging: {exc}", err=True)
            # (Continue without raising, as logging is non-critical)

    # ------------------------------------------------------------------ 
    # 4) Learning (training loop) 
    # ------------------------------------------------------------------ 
    # Filter hyperparams for model.learn() arguments (e.g., total_timesteps, progress_bar, etc.).
    learn_cfg = filter_config(model.learn, hyperparams)
    try:
        typer.echo(f"[INFO] learning parameters: {learn_cfg}")
        # Begin training the model with the specified number of timesteps and other parameters.
        model.learn(**learn_cfg)
    except Exception as exc:
        # Catch any runtime exceptions during training (e.g., out-of-memory) and log them.
        typer.echo(f"[ERROR] Learning failed: {exc}", err=True)

    # ------------------------------------------------------------------ 
    # 5) Save and tidy up 
    # ------------------------------------------------------------------ 
    env.close()  # Ensure the environment is closed and resources are freed.
    try:
        # Construct a filename for the saved model using the algorithm name and environment ID.
        _algo = algo_cls.__name__.lower()                 # e.g., "ppo"
        _env_id = vec_env_kwargs.get("env_id", "")
        model_path = f"{_algo}_{_env_id}.zip"
        model.save(model_path)
        typer.echo(f"[OK] Model saved to {model_path}")
    except Exception as exc:
        typer.echo(f"[ERROR] Failed to save model: {exc}", err=True)

    return  # Function returns None (model is saved to disk)

# --------------------------------------------------------------------------- 
# Stand-alone usage (quick smoke test) 
# --------------------------------------------------------------------------- 
if __name__ == "__main__":  # pragma: no cover
    # If this module is run directly, perform a test training with default hyperparams.
    train_sb3({})
